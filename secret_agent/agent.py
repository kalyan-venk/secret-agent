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


MAX_ARG_ECHO = 80


def render_call(call) -> str:
    """A compact record of a call, to stand in for the raw JSON in history.

    Long values are cut -- a write_file carrying 4KB of content would
    otherwise put that 4KB into history a second time, on top of the copy
    already in the tool result.
    """
    bits = []
    for k, v in call.arguments.items():
        s = repr(v)
        if len(s) > MAX_ARG_ECHO:
            s = s[:MAX_ARG_ECHO] + "...'"
        bits.append(f"{k}={s}")
    return f"[calling {call.name}({', '.join(bits)})]"


class AgentFailure(RuntimeError):
    """The agent did not produce an answer.

    Deliberately an exception rather than a returned string. Both subclasses
    below mean "this run failed", and handing back the last partial text
    instead would let a caller mistake a failure for an answer -- which is
    exactly how a bad number ends up in an aggregate and then in a report.

    The partial conversation and step list hang off the exception so a caller
    that wants to inspect or salvage can, explicitly.
    """

    def __init__(self, msg: str, conversation: Conversation, steps: list["Step"]):
        super().__init__(msg)
        self.conversation = conversation
        self.steps = steps


class AgentLoopExhausted(AgentFailure):
    pass


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

            # What goes into history when the model both talked AND called a
            # tool? See record_assistant_turn below -- it's a real decision
            # with a real failure mode either way.
            self.conversation.add_assistant(
                self.record_assistant_turn(completion, parsed)
            )

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

    def record_assistant_turn(self, completion: Completion, parsed: ParseResult) -> str:
        """Decide what text to store in history for a turn that contained a
        tool call. Return the string to record as the assistant message.

        ## Why this is a decision and not a detail

        Real transcript, iteration 1, llama3.1:8b, verbatim:

            {"name": "echo", "arguments": {"text": "pineapple"}}

            The echo tool returned: "pineapple"

        The tool had not run yet. The model emitted the call and then narrated
        the result it *expected*, in the same message. It guessed right here
        because echo is trivial. For read_file it would be inventing file
        contents into its own history -- and on the next turn it cannot tell
        its guess apart from the real tool message underneath it.

        ## What you have to work with

            completion.text   the raw text, tool-call JSON and all
            parsed.text       the prose with the JSON blocks removed
            parsed.calls      list[ToolCall] -- .name, .arguments

        ## I reasoned my way to the wrong answer, then measured

        My argument was: text BEFORE a call is real reasoning, worth keeping;
        text AFTER a call is narration of a result that does not exist yet, so
        drop it; and while we're here, replace the raw JSON with a compact
        `[calling echo(text='hi')]` line to save context.

        The reasoning is fine. The last clause was wrong, and it cost a 25%
        increase in model calls. `scripts/turn_policy.py`, 8 tasks,
        llama3.1:8b:

            policy                  done  failed  avg iters  tool calls
            1 raw text                 6       0        2.2           9
            2 prose only               5       1        4.3          24
            3 call line only           6       0        2.8          13
            4 before + call line       6       0        2.7          12
            5 before + call + after    6       0        2.7          12
            6 before + raw json        8       0        2.1          13

        Two things fall out of that, and neither was obvious from the armchair:

        **Dropping prose_after is free.** Policies 4 and 5 differ only in
        whether the trailing narration is kept, and they are identical on
        every column. The fabrication isn't load-bearing.

        **Paraphrasing the call is not free.** 1 vs 4 differ only in raw JSON
        versus my tidy call line, and the tidy version costs half an iteration
        and three extra tool calls per task. Policy 2, which removes the call
        record entirely, is worst by a mile -- one outright failure and nearly
        three times the tool calls, because the model loses the evidence that
        it already called something and calls it again.

        The model appears to need its own call back **in the format it emitted
        it**. A paraphrase is close enough for a human reading a transcript and
        apparently not close enough for an 8B model reading its own history.

        ## Chosen: 6

        Prose before the call, then the model's own JSON verbatim, then drop
        the trailing narration. Ties the best policy on every measured column
        while removing the fabricated result. Strictly dominant, so there is
        no trade-off to argue about.

        ## Caveats

        Eight tasks on one model. The 1-vs-4 gap is half an iteration, which
        is not much. What I'd actually defend is the *ordering* -- policy 2 is
        clearly bad and policy 6 is never worse than 1 -- not the decimals.

        The remaining cost is real but small: a genuine trailing thought
        ("...and if that fails I'll try the backup path") is lost with the
        narration, since nothing distinguishes them syntactically.
        """
        if not parsed.calls:
            return completion.text

        parts = []
        if parsed.prose_before:
            parts.append(parsed.prose_before)

        # The model's own JSON, verbatim -- not render_call(). See above:
        # paraphrasing this measurably costs iterations.
        #
        # Deduped, because when ONE span parses to an array of N calls, every
        # ToolCall.raw is that same whole span (parsing.py sets raw to the
        # enclosing text, not the individual element). Extending naively wrote
        # the array into history N times -- token bloat plus N duplicate call
        # records for the model to read. Caught in review 2026-07-25.
        seen = set()
        for c in parsed.calls:
            if c.raw and c.raw not in seen:
                seen.add(c.raw)
                parts.append(c.raw)

        # parsed.prose_after is dropped. Measured free; removes the fabrication.
        return "\n".join(parts)

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
