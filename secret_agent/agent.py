"""The loop.

In one sentence: the model asks for a tool, my code runs it, I put the result
back into the conversation, and that repeats until the model stops asking.

Everything else in this repo is scaffolding for the twenty-odd lines in
Agent._step. That's not modesty, it's the actual shape of an agent -- the
loop is trivial and all the difficulty is in the four places it can go wrong:

    runaway          -> max_iterations, hard cap, raises
    malformed call   -> parsing.py + bounded retry, below
    tool blows up    -> caught in registry.execute, comes back as a result
    context overflow -> context.py, checked before every model call

Those four are the whole interview answer. If you can name them and say what
you did about each, you understand this file.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config
from .context import ContextManager
from .conversation import Conversation
from .llm import Completion, LLMClient, OllamaClient
from .parsing import STATS, ParseResult, parse_native_tool_calls, parse_tool_calls
from .prompts import DEFAULT_BASE, RETRY_NUDGE, build_system_prompt
from .tools.base import ToolResult
from .tools.registry import Registry


class AgentFailure(RuntimeError):
    """The agent did not produce an answer.

    Deliberately an exception rather than a returned string. Both subclasses
    below mean "this run failed", and handing back the last partial text
    instead would let a caller mistake a failure for an answer -- which is
    exactly how a bad number ends up in a metric and then on a resume.

    The partial conversation and step list hang off the exception so a caller
    that wants to inspect or salvage can, explicitly.
    """

    def __init__(self, msg: str, conversation: Conversation, steps: list["Step"]):
        super().__init__(msg)
        self.conversation = conversation
        self.steps = steps


class AgentLoopExhausted(AgentFailure):
    """Hit max_iterations without the model ever stopping."""


class ParseRetriesExhausted(AgentFailure):
    """The model kept emitting things that looked like tool calls and weren't.

    Split from AgentLoopExhausted because the fix is different: a runaway loop
    usually means the task is underspecified, this usually means the prompt
    format isn't landing or the model is too small for the schema.
    """


@dataclass
class Step:
    """One trip through the loop. Enough to reconstruct what happened without
    a real tracing stack -- proper span-level observability is out of scope
    here on purpose."""

    n: int
    completion_text: str = ""
    calls: list[str] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    repaired: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    compacted: bool = False
    elapsed_s: float = 0.0


@dataclass
class AgentRun:
    answer: str
    steps: list[Step]
    conversation: Conversation

    @property
    def iterations(self) -> int:
        return len(self.steps)

    @property
    def tool_calls(self) -> int:
        return sum(len(s.calls) for s in self.steps)

    @property
    def repair_rate(self) -> float:
        """Fraction of tool-emitting steps whose JSON needed fixing.

        Report this next to any result you compare across models. A run where
        the parser did 40% of the work and one where it did 2% are not
        comparable, and the gap looks like model quality if you don't print it.
        """
        with_calls = [s for s in self.steps if s.calls]
        if not with_calls:
            return 0.0
        return sum(1 for s in with_calls if s.repaired) / len(with_calls)

    def summary(self) -> str:
        return (
            f"{self.iterations} iterations, {self.tool_calls} tool calls, "
            f"repair rate {self.repair_rate:.0%}, "
            f"{sum(s.prompt_tokens for s in self.steps)} prompt tokens total"
        )


class Agent:
    def __init__(
        self,
        registry: Registry,
        client: LLMClient | None = None,
        cfg: Config | None = None,
        system: str | None = None,
        on_step: Callable[[Step], None] | None = None,
    ):
        self.cfg = cfg or Config.from_env()
        self.client = client or OllamaClient(self.cfg)
        self.registry = registry
        self.on_step = on_step
        self.ctx = ContextManager(self.cfg, self.client)

        base = system or DEFAULT_BASE
        if self.cfg.tool_mode == "prompted":
            sys_prompt = build_system_prompt(base, registry.prompt_block())
        else:
            # native mode hands schemas to the server separately, so putting
            # them in the prompt too just doubles the token cost
            sys_prompt = base

        self.conversation = Conversation(sys_prompt)

    # -----------------------------------------------------------------

    def run(self, task: str) -> AgentRun:
        self.conversation.add_user(task)
        steps: list[Step] = []
        parse_failures = 0

        for i in range(1, self.cfg.max_iterations + 1):
            step = Step(n=i)
            t0 = time.perf_counter()

            # Check the budget BEFORE the call, not after. Finding out you
            # overflowed by reading a truncated reply is too late -- and with
            # Ollama you don't even get an error, the front of the history
            # just silently isn't there any more.
            step.compacted = self.ctx.ensure_fits(self.conversation)

            completion = self.client.complete(
                self.conversation.to_wire(),
                tools=self.registry.schemas() if self.cfg.tool_mode == "native" else None,
            )
            step.prompt_tokens = completion.usage.prompt_tokens
            step.completion_tokens = completion.usage.completion_tokens
            step.completion_text = completion.text

            parsed = self._parse(completion)

            # --- stop condition ------------------------------------
            # No tool calls => the model is done talking. Note this keys off
            # the ABSENCE OF A CALL, not the presence of prose. Small models
            # narrate constantly ("Let me look at that file.") while also
            # emitting the call; stopping on prose would end every run on
            # iteration 1.
            if not parsed.calls:
                if parsed.problems:
                    # It tried to call something and produced garbage. Hand
                    # the error back and let it try again -- bounded, because
                    # a model that has failed twice is not going to get it on
                    # the third go, it's going to loop and bill you for it.
                    parse_failures += 1
                    step.elapsed_s = time.perf_counter() - t0
                    steps.append(step)
                    self._emit(step)

                    if parse_failures > self.cfg.max_parse_retries:
                        # Do NOT fall through and return completion.text here.
                        # That text is broken JSON; returning it as "the
                        # answer" dresses a failure up as a result. Same
                        # reasoning as the iteration cap.
                        raise ParseRetriesExhausted(
                            f"model produced unparseable tool calls "
                            f"{parse_failures} times in a row. Last: "
                            f"{completion.text[:200]!r}",
                            self.conversation,
                            steps,
                        )

                    self.conversation.add_assistant(completion.text)
                    self.conversation.add_user(
                        RETRY_NUDGE.format(problem="; ".join(parsed.problems)[:400])
                    )
                    continue

                self.conversation.add_assistant(completion.text)
                step.elapsed_s = time.perf_counter() - t0
                steps.append(step)
                self._emit(step)
                return AgentRun(
                    answer=parsed.text or completion.text,
                    steps=steps,
                    conversation=self.conversation,
                )

            parse_failures = 0  # it recovered; don't hold the old failures against it
            step.calls = [c.name for c in parsed.calls]
            step.repaired = any(c.repairs for c in parsed.calls)

            # Keep the prose the model wrote alongside the call. It's often
            # the reasoning ("I need to see the config first") and dropping it
            # makes the transcript unreadable later.
            self.conversation.add_assistant(completion.text)

            # --- execute -------------------------------------------
            # All calls from one completion are executed and ALL their results
            # go back in one turn. Splitting them across turns trains the
            # model out of batching, which costs a round trip every time it
            # would have asked for two things at once.
            #
            # Sequential, not threaded. These are file reads on a laptop; the
            # 6-second model call dominates and concurrency here would buy
            # nothing while making tool side effects race.
            for c in parsed.calls:
                result = self.registry.execute(c)
                result = self.ctx.trim_tool_result(result)
                step.results.append(result)
                self.conversation.add_tool_result(c.id, c.name, result.for_model())

            step.elapsed_s = time.perf_counter() - t0
            steps.append(step)
            self._emit(step)

        raise AgentLoopExhausted(
            f"gave up after {self.cfg.max_iterations} iterations without a final answer. "
            f"Last text was: {steps[-1].completion_text[:200]!r}",
            self.conversation,
            steps,
        )

    # -----------------------------------------------------------------

    def _parse(self, completion: Completion) -> ParseResult:
        if self.cfg.tool_mode == "native" and completion.native_tool_calls:
            return parse_native_tool_calls(completion.native_tool_calls)
        return parse_tool_calls(completion.text, known_names=self.registry.names)

    def _emit(self, step: Step) -> None:
        if self.on_step:
            self.on_step(step)
        if self.cfg.verbose:
            print(f"  [{step.n}] calls={step.calls or '-'} "
                  f"repaired={step.repaired} tok={step.prompt_tokens} "
                  f"{step.elapsed_s:.1f}s")

    def parse_stats(self) -> str:
        return STATS.summary()
