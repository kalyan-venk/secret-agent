"""Context management.

The problem, precisely: the API is stateless, so every turn re-sends the whole
history, so the payload grows monotonically while the window does not. Tool
results are the fast-growing part -- a grep can add more tokens in one call
than twenty turns of conversation.

## What actually breaks when you exceed the window

I guessed at this first and got it wrong, so here is the measurement instead.
Sent a 3,580-token history to llama3.1:8b at three window sizes, with a
distinct marker in the system prompt, the first user message, and the last
(scripts/overflow_probe.py, 2026-07-25):

    num_ctx   prompt_eval   system   first user   last user
    8192           3580       kept        kept        kept
    1024             53       kept       DROPPED      kept
     256             53       kept       DROPPED      kept

Every response was HTTP 200 with no `error` field.

So it is not "drops from the front", which is what I assumed. Ollama keeps
the system prompt and the most recent message and discards the middle -- it
performs a crude truncation on your behalf and does not tell you.

That is worse than an error in a specific way. The system prompt surviving is
exactly the thing that makes it hard to notice: the model still behaves like
itself, still follows its instructions, still sounds fine. It has just
forgotten the middle of the task. 3,580 tokens became 53 and the only visible
symptom was an answer that left something out.

For comparison, hosted APIs are kinder about this -- Anthropic and OpenAI
both return a 400 with an explicit token count. On Ollama there is no
downstream signal at all, so counting before you send is the only defense.
That is why ensure_fits() is called before the request rather than after.

## Two strategies, implemented so they can be compared

  truncate   drop whole turns from the oldest end, keep system + last N.
             Free, instant, deterministic. Loses everything it drops.
  summarize  same drop, but a model call first turns the dropped span into a
             paragraph that goes back in as context. Costs one extra
             round-trip and the summary is lossy in a sneakier way than
             truncation is -- truncation loses facts visibly, summarization
             replaces "edit config.py line 40" with "discussed configuration"
             and you cannot tell from reading it that a filename was lost.

Neither is right in general. Truncation is right when old turns are
genuinely done with; summarization is right when the task has a thread
running through it. The measurable difference is in tests and in the
`compare_strategies` helper at the bottom.

## A note on prompt caching

The system prompt sits at the front of every payload. Both Ollama's KV cache
and hosted prompt caching key on a prefix match, so editing the system prompt
mid-conversation invalidates everything from the first differing token
onward -- you pay full processing for the whole history again.

Which is exactly what naive compaction does if you're careless: rewriting
history at the front busts the cache for every subsequent turn. That's a real
argument for compacting rarely and in large chunks rather than trimming a
little every turn, and it's why compaction_headroom exists instead of
compacting the moment we cross the line.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .conversation import Conversation, Message
from .tools.base import ToolResult

# Per-message overhead from the chat template: role headers, BOS, the
# assistant turn marker. MEASURED, not guessed: send the empty string to
# llama3.1:8b and prompt_eval_count comes back as 10. See
# scripts/calibrate_tokens.py.
#
# I had 4 in here first, from memory of an OpenAI figure. It's 10 for this
# template, and on a 40-message history that's a 240-token error, which is
# 3% of an 8k window spent on a number I didn't check.
PER_MESSAGE_OVERHEAD = 10


class TokenCounter:
    """Counts tokens, exactly when it can afford to and approximately when it
    can't.

    The real count means a round trip to the server per string. Doing that on
    every message on every turn would cost more time than the generation. So:
    exact counts are cached per message (messages never change once added),
    and the cheap heuristic is used for anything transient.

    For a hosted Claude model the equivalent exact call is the count_tokens
    endpoint. tiktoken is NOT the right tool there -- it's OpenAI's tokenizer
    and undercounts Claude noticeably. Using it and calling the result "the
    token count" is a quiet way to blow a budget.
    """

    def __init__(self, client=None, exact: bool = False):
        self.client = client
        self.exact = exact and hasattr(client, "count_tokens")
        self._cache: dict[str, int] = {}
        self.exact_calls = 0

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if not self.exact:
            return approx(text)
        key = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
        if key not in self._cache:
            self.exact_calls += 1
            self._cache[key] = self.client.count_tokens(text)
        return self._cache[key]

    def count_message(self, m: Message) -> int:
        return self.count_text(m.content) + PER_MESSAGE_OVERHEAD

    def count(self, conv: Conversation) -> int:
        return sum(self.count_message(m) for m in conv.messages)


# Token estimation, second attempt.
#
# ## Why there was a first attempt
#
# The original version bucketed on the density of eight characters ({}[]":,\)
# and claimed in this very comment to be "chosen to sit BELOW every measured
# value so the estimate errs toward overcounting". An external review checked
# that claim against the real tokenizer and it was false:
#
#     content                approx   real   ratio
#     code with operators       100    190   0.53   <- 47% UNDER
#     digits                    102    220   0.46   <- 54% UNDER
#     file paths / URLs         118    130   0.91   <- 9%  UNDER
#     emoji                      18    224   0.08   <- 92% UNDER
#     json (was calibrated)      96     85   1.13   ok
#     prose (was calibrated)    104    101   1.03   ok
#
# It was safe on precisely the two content types I calibrated it against, and
# badly unsafe everywhere else. `(a+b)*(c-d)` scores ~0% density because none
# of `()+*-=` were in the set, so it got the prose divisor. Code is the most
# common content in an agent history, so the estimator was worst exactly where
# it mattered, and it fails toward believing there is room when there isn't --
# which silently overflows the window, the failure this whole module exists to
# prevent.
#
# ## What replaced it
#
# Per-character-class weights instead of one global divisor, because different
# character classes genuinely tokenize at different rates and no single divisor
# can span 0.36 chars/token (emoji) to 4.7 (prose). Each weight is set ABOVE
# the measured rate so the estimate errs high.
#
#     class                 measured tokens/char   weight used
#     letters and spaces           0.22               0.27
#     digits                       0.54               0.60
#     ASCII punctuation            ~0.79              0.85
#     non-ASCII BMP (CJK)          0.87               1.10
#     astral plane (emoji)         2.80               3.40
#
# Overcounting compacts slightly early, which costs a summarization call.
# Undercounting silently truncates the prompt. Those are not symmetric, so
# every weight rounds up.
_W_LETTER = 0.27
_W_DIGIT = 0.60
_W_PUNCT = 0.85
_W_WIDE = 1.10
_W_ASTRAL = 3.40


def approx(text: str) -> int:
    """Cheap token estimate, biased high -- and this time verified to be.

    `tests/test_context.py::test_approx_never_undercounts_any_content_type`
    pins the property that the previous version claimed and did not have.

    When the number has to be right, use TokenCounter(exact=True): it asks the
    server to tokenize and is the model's own count. This is for the thousands
    of counts per run where a round trip each would cost more than generation.
    """
    if not text:
        return 0

    total = 0.0
    for ch in text:
        o = ord(ch)
        if o > 0xFFFF:
            total += _W_ASTRAL      # emoji and friends, ~2.8 tokens each
        elif o > 0x7F:
            total += _W_WIDE        # CJK etc
        elif ch.isalpha() or ch.isspace():
            total += _W_LETTER
        elif ch.isdigit():
            total += _W_DIGIT
        else:
            total += _W_PUNCT       # every operator, bracket, quote, slash
    return max(1, int(total) + 1)


@dataclass
class Compaction:
    """What happened, so it can be printed and asserted on."""

    fired: bool = False
    strategy: str = ""
    before: int = 0
    after: int = 0
    dropped_turns: int = 0
    dropped_messages: int = 0
    summary: str = ""
    cost_ms: float = 0.0

    @property
    def saved(self) -> int:
        return self.before - self.after

    def __str__(self) -> str:
        if not self.fired:
            return "no compaction"
        return (
            f"compacted ({self.strategy}): {self.before} -> {self.after} tokens "
            f"(-{self.saved}, {100 * self.saved / max(1, self.before):.0f}%), "
            f"{self.dropped_turns} turns dropped, {self.cost_ms:.0f}ms"
        )


class ContextManager:
    def __init__(self, cfg: Config, client=None, exact_tokens: bool = False):
        self.cfg = cfg
        self.client = client
        self.counter = TokenCounter(client, exact=exact_tokens)
        self.history: list[Compaction] = []
        self.spill_dir = Path(cfg.root) / ".spill"

    # -----------------------------------------------------------------

    def usage(self, conv: Conversation) -> tuple[int, int]:
        return self.counter.count(conv), self.cfg.history_budget

    def ensure_fits(self, conv: Conversation) -> bool:
        """Compact if needed. Returns whether it fired.

        Called before every model request. The headroom check means we
        compact at 85% rather than at 100%: partly so there's slack for the
        next turn's tool result, partly because compacting in one big chunk
        busts the prompt cache once instead of every turn.
        """
        used, budget = self.usage(conv)
        threshold = int(budget * self.cfg.compaction_headroom)

        if used <= threshold:
            return False

        t0 = time.perf_counter()
        if self.cfg.strategy == "truncate":
            c = self._truncate(conv, used)
        elif self.cfg.strategy == "summarize":
            c = self._summarize(conv, used)
        else:
            raise ValueError(f"unknown strategy {self.cfg.strategy!r}")

        c.cost_ms = (time.perf_counter() - t0) * 1000
        c.after = self.counter.count(conv)
        conv.compactions += 1
        self.history.append(c)

        if self.cfg.verbose:
            print(f"  [context] {c}")

        # If a single turn is bigger than the whole budget, no strategy that
        # drops whole turns can help. Say so instead of looping silently --
        # this is a real state and the fix is trim_tool_result, not more
        # compaction.
        if c.after > budget:
            print(
                f"  [context] WARNING: still {c.after} tokens after compaction, "
                f"budget is {budget}. A single turn is too large to fit; the "
                f"model will see a silently truncated prompt."
            )

        return True

    # -----------------------------------------------------------------

    def _plan_drop(self, conv: Conversation) -> tuple[list[list[Message]], list[list[Message]]]:
        """Split turns into (drop, keep).

        Keeps the last keep_recent_turns. The system message isn't in turns()
        at all, so it is never a candidate -- that's structural rather than a
        special case, which is why turns() excludes it.
        """
        turns = conv.turns()
        keep_n = max(1, self.cfg.keep_recent_turns)
        if len(turns) <= keep_n:
            return [], turns
        return turns[:-keep_n], turns[-keep_n:]

    def _rebuild(self, conv: Conversation, keep: list[list[Message]], prefix: Message | None):
        system = [m for m in conv.messages if m.role == "system"]
        rest = [m for t in keep for m in t]
        conv.messages = system + ([prefix] if prefix else []) + rest

    def _truncate(self, conv: Conversation, before: int) -> Compaction:
        drop, keep = self._plan_drop(conv)
        if not drop:
            return Compaction(fired=False, strategy="truncate", before=before)

        # A marker, so the model knows history is missing rather than
        # concluding the conversation started where it now appears to. Without
        # this llama3.1 would re-introduce itself mid-task.
        n_msgs = sum(len(t) for t in drop)
        marker = Message(
            role="user",
            content=f"[{n_msgs} earlier messages were dropped to stay within the "
                    f"context window. If you need something from them, ask.]",
        )
        self._rebuild(conv, keep, marker)

        return Compaction(
            fired=True, strategy="truncate", before=before,
            dropped_turns=len(drop), dropped_messages=n_msgs,
        )

    def _summarize(self, conv: Conversation, before: int) -> Compaction:
        drop, keep = self._plan_drop(conv)
        if not drop:
            return Compaction(fired=False, strategy="summarize", before=before)

        n_msgs = sum(len(t) for t in drop)
        transcript = "\n".join(
            f"{m.role}: {m.content[:1500]}" for t in drop for m in t
        )

        summary = self._ask_for_summary(transcript)

        marker = Message(
            role="user",
            content=(
                f"[Summary of {n_msgs} earlier messages, which have been dropped "
                f"to stay within the context window:]\n\n{summary}"
            ),
        )
        self._rebuild(conv, keep, marker)

        return Compaction(
            fired=True, strategy="summarize", before=before,
            dropped_turns=len(drop), dropped_messages=n_msgs, summary=summary,
        )

    def _ask_for_summary(self, transcript: str) -> str:
        if self.client is None:
            return "(no client available to summarize; history was dropped)"

        # A separate one-shot call, NOT appended to the conversation. Putting
        # the summarization request into the history being summarized is a
        # loop you notice about an hour in.
        prompt = (
            "Summarize this conversation excerpt for an assistant that will "
            "continue the task. Keep concrete details -- filenames, exact "
            "values, decisions made, things already tried and their outcome. "
            "Drop pleasantries. Under 200 words. No preamble.\n\n"
            f"---\n{transcript}\n---"
        )
        try:
            out = self.client.complete([{"role": "user", "content": prompt}])
            return out.text.strip() or "(summarizer returned nothing)"
        except Exception as e:
            # Falling back to truncation is the right failure: the run
            # continues, degraded and visibly so, rather than dying because
            # a helper call timed out.
            return f"(summarization failed: {type(e).__name__}; history was dropped)"

    # -----------------------------------------------------------------

    def trim_tool_result(self, result: ToolResult) -> ToolResult:
        """Keep a big tool result out of history without losing it.

        A grep over a real repo returns thousands of lines. Putting that in
        history costs the window for every subsequent turn, forever, for
        something the model needed once.

        Head and tail, not head alone: for structured output the interesting
        part is often at the end (the summary line, the last error), and
        head-only truncation reliably cuts it off.

        The full result goes to .spill/ and the marker names the file, so the
        model can read_file it if it turns out to matter. That's the bit that
        makes this lossless-in-principle rather than just lossy-and-hoping.
        """
        limit = self.cfg.max_tool_result_chars
        if len(result.content) <= limit:
            return result

        path = self._spill(result)
        head = result.content[: limit // 2]
        tail = result.content[-limit // 2:]
        cut = len(result.content) - limit

        result.content = (
            f"{head}\n\n"
            f"... [{cut} characters trimmed to fit the context window. "
            f"Full output saved to {path} -- read_file it if you need the middle.] ...\n\n"
            f"{tail}"
        )
        result.truncated = True
        result.spill_path = str(path)
        return result

    def _spill(self, result: ToolResult) -> str:
        try:
            self.spill_dir.mkdir(parents=True, exist_ok=True)
            name = f"{result.name}_{result.call_id}.txt"
            (self.spill_dir / name).write_text(result.content, encoding="utf-8")
            return f".spill/{name}"
        except OSError:
            return "(spill failed; middle of output is gone)"

    # -----------------------------------------------------------------

    def report(self) -> str:
        if not self.history:
            return "context: no compaction needed"
        lines = [f"context: {len(self.history)} compactions"]
        for c in self.history:
            lines.append("  " + str(c))
        if self.counter.exact_calls:
            lines.append(f"  ({self.counter.exact_calls} exact token counts)")
        return "\n".join(lines)


def compare_strategies(conv: Conversation, cfg: Config, client=None) -> str:
    """Run both strategies on a copy of the same conversation and print the
    difference. This is the thing to run before claiming one is better --
    'summarization is better' is an assumption until you look at what each
    one actually kept.
    """
    import copy

    rows = []
    for strat in ("truncate", "summarize"):
        c2 = copy.deepcopy(conv)
        cfg2 = Config(**{**cfg.__dict__, "strategy": strat})
        cm = ContextManager(cfg2, client)
        before, budget = cm.usage(c2)
        fired = cm.ensure_fits(c2)
        after, _ = cm.usage(c2)
        cost = cm.history[-1].cost_ms if cm.history else 0.0
        rows.append((strat, before, after, fired, cost, len(c2.messages)))

    out = [f"budget {cfg.history_budget} tokens", ""]
    out.append(f"{'strategy':<12}{'before':>8}{'after':>8}{'saved':>8}{'msgs':>7}{'ms':>9}")
    for strat, b, a, fired, cost, nmsg in rows:
        out.append(f"{strat:<12}{b:>8}{a:>8}{b - a:>8}{nmsg:>7}{cost:>9.0f}")
    return "\n".join(out)
