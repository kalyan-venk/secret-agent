import copy

import pytest

from secret_agent.config import Config
from secret_agent.context import (
    PER_MESSAGE_OVERHEAD,
    Compaction,
    ContextManager,
    TokenCounter,
    approx,
    compare_strategies,
)
from secret_agent.conversation import Conversation
from secret_agent.llm import Completion
from secret_agent.tools.base import ToolResult


class FakeSummarizer:
    def __init__(self, text="Earlier: user asked about config.py, we edited line 40."):
        self.text = text
        self.calls = 0

    def complete(self, messages, tools=None):
        self.calls += 1
        return Completion(text=self.text)

    def count_tokens(self, text):
        return approx(text)


def big_conversation(turns=20, chars=400) -> Conversation:
    c = Conversation("system prompt, pinned, never dropped")
    for i in range(turns):
        c.add_user(f"turn {i}: " + ("word " * (chars // 5)))
        c.add_assistant(f"reply {i}: " + ("word " * (chars // 5)))
    return c


def cfg(**kw):
    base = dict(num_ctx=2000, response_reserve_tokens=200, keep_recent_turns=3,
                compaction_headroom=0.85)
    base.update(kw)
    return Config(**base)


# --- token counting ---------------------------------------------------


def test_approx_overcounts_rather_than_undercounts_on_json():
    # the measured failure mode of chars/4: it undercounts JSON by ~29%,
    # which is the direction that silently overflows a budget
    blob = '{"name": "read_file", "arguments": {"path": "src/main.py"}}' * 5
    naive = len(blob) // 4
    assert approx(blob) > naive


# real token counts from llama3.1:8b, measured 2026-07-25. Pinned here rather
# than fetched, so the property is checked offline on every run.
_MEASURED = [
    ("code with operators", "result = (a+b)*(c-d)/(e+f) if x>y else g<h\n" * 10, 190),
    ("file paths", "/usr/local/lib/python3.12/site-packages/pkg/mod.py\n" * 10, 130),
    ("digits", "1234567890 9876543210 5551234567 42 3.14159\n" * 10, 220),
    ("emoji", "\U0001f525\U0001f680\u2728\U0001f3af\U0001f4a1" * 16, 224),
    ("cjk", "\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5\u6587\u6863\u7528\u6765\u68c0\u67e5\u5206\u8bcd\u5668\u7684\u884c\u4e3a\u65b9\u5f0f" * 5, 70),
    ("json", '{"name": "read_file", "arguments": {"path": "a.py"}}' * 5, 85),
    ("prose", "The quick brown fox jumps over the lazy dog. " * 10, 101),
    ("markdown table", "| col | value | note |\n|---|---|---|\n| a | 1 | x |\n" * 8, 176),
    ("tool output", "src/module_9.py:27: TODO handle this before release\n" * 10, 140),
    ("code and prose", "def f(x):\n    # returns double\n    return x*2\n" * 10, 150),
    ("yaml", "name: build\non:\n  push:\n    branches: [main]\n  pull_request:\n" * 8, 152),
]


@pytest.mark.parametrize("label,text,real", _MEASURED, ids=[m[0] for m in _MEASURED])
def test_approx_never_undercounts_any_content_type(label, text, real):
    """The property the previous estimator CLAIMED in a comment and did not have.

    It undercounted code by 47% and emoji by 92% while a comment directly
    above it asserted the opposite; the full table and the retune are in
    context.py. What matters here is that a claim about all inputs belongs in
    a test, where it gets checked, and not in a comment, where it doesn't.
    See MISTAKES.md #15.
    """
    assert approx(text) >= real, (
        f"{label}: estimated {approx(text)} but the real count is {real} -- "
        "undercounting silently overflows the context window"
    )


def test_approx_does_not_overcount_absurdly():
    # erring high is safe, but erring 5x high means compacting constantly and
    # paying for summarization nothing needed
    for label, text, real in _MEASURED:
        assert approx(text) <= real * 2.0, f"{label}: {approx(text)} vs {real}"


def test_approx_is_close_enough_on_prose():
    prose = "The quick brown fox jumps over the lazy dog. " * 20
    naive = len(prose) // 4
    # within 15% of the naive rule for prose, where the naive rule is decent
    assert 0.85 * naive <= approx(prose) <= 1.2 * naive


def test_per_message_overhead_is_the_measured_value():
    # 10, from scripts/calibrate_tokens.py against llama3.1's chat template.
    # Pinned so that anyone who wants a different number has to go re-measure.
    assert PER_MESSAGE_OVERHEAD == 10


def test_counter_caches_exact_calls():
    class Counting:
        def __init__(self):
            self.n = 0

        def count_tokens(self, t):
            self.n += 1
            return len(t) // 4

    client = Counting()
    tc = TokenCounter(client, exact=True)
    for _ in range(5):
        tc.count_text("the same string every time")
    assert client.n == 1
    assert tc.exact_calls == 1


def test_counter_falls_back_to_approx_without_a_client():
    tc = TokenCounter(None, exact=True)
    assert tc.count_text("hello world") == approx("hello world")


# --- when it fires ----------------------------------------------------


def test_small_conversation_is_left_alone():
    cm = ContextManager(cfg(), FakeSummarizer())
    c = Conversation("sys")
    c.add_user("hi")
    assert cm.ensure_fits(c) is False
    assert len(c.messages) == 2


def test_it_fires_at_the_headroom_threshold_not_at_the_limit():
    # compacting at 100% leaves no slack for the next tool result, and
    # compacting a little every turn busts the prompt cache every turn
    c = cfg(compaction_headroom=0.85)
    cm = ContextManager(c, FakeSummarizer())
    conv = big_conversation()
    used, budget = cm.usage(conv)
    assert used > budget * 0.85
    assert cm.ensure_fits(conv) is True


# --- truncation -------------------------------------------------------


def test_truncate_drops_oldest_and_keeps_recent():
    cm = ContextManager(cfg(strategy="truncate"))
    conv = big_conversation()
    before = len(conv.messages)

    assert cm.ensure_fits(conv) is True
    c = cm.history[0]

    assert c.strategy == "truncate"
    assert c.after < c.before
    assert len(conv.messages) < before
    # the most recent turns are still there verbatim
    assert "turn 19" in conv.messages[-2].content


def test_system_prompt_survives_every_strategy():
    for strat in ("truncate", "summarize"):
        cm = ContextManager(cfg(strategy=strat), FakeSummarizer())
        conv = big_conversation()
        cm.ensure_fits(conv)
        assert conv.messages[0].role == "system"
        assert conv.messages[0].content == "system prompt, pinned, never dropped"


def test_truncation_leaves_a_marker_so_the_model_knows_it_lost_something():
    # without this llama3.1 concludes the conversation started where it now
    # appears to and re-introduces itself mid-task
    cm = ContextManager(cfg(strategy="truncate"))
    conv = big_conversation()
    cm.ensure_fits(conv)
    assert any("were dropped" in m.content for m in conv.messages)


def test_truncate_costs_no_model_calls():
    client = FakeSummarizer()
    cm = ContextManager(cfg(strategy="truncate"), client)
    cm.ensure_fits(big_conversation())
    assert client.calls == 0


# --- summarization ----------------------------------------------------


def test_summarize_costs_exactly_one_extra_model_call():
    client = FakeSummarizer()
    cm = ContextManager(cfg(strategy="summarize"), client)
    cm.ensure_fits(big_conversation())
    assert client.calls == 1


def test_summary_text_replaces_the_dropped_span():
    client = FakeSummarizer()
    cm = ContextManager(cfg(strategy="summarize"), client)
    conv = big_conversation()
    cm.ensure_fits(conv)
    assert any("config.py" in m.content for m in conv.messages)
    assert cm.history[0].summary


def test_the_summarization_request_is_not_added_to_the_history_it_summarizes():
    # doing that is a loop you notice about an hour in
    client = FakeSummarizer()
    cm = ContextManager(cfg(strategy="summarize"), client)
    conv = big_conversation()
    cm.ensure_fits(conv)
    assert not any("Summarize this conversation excerpt" in m.content
                   for m in conv.messages)


def test_a_failing_summarizer_degrades_instead_of_killing_the_run():
    class Broken:
        def complete(self, *a, **k):
            raise TimeoutError("model took too long")

        def count_tokens(self, t):
            return approx(t)

    cm = ContextManager(cfg(strategy="summarize"), Broken())
    conv = big_conversation()
    assert cm.ensure_fits(conv) is True
    assert any("summarization failed" in m.content for m in conv.messages)


def test_unknown_strategy_is_a_loud_error():
    cm = ContextManager(cfg(strategy="vibes"), FakeSummarizer())
    with pytest.raises(ValueError, match="vibes"):
        cm.ensure_fits(big_conversation())


# --- comparing the two ------------------------------------------------


def test_both_strategies_free_real_tokens_and_can_be_compared():
    """The 'done when' for phase 5: print before/after at the moment it fires."""
    conv = big_conversation()
    client = FakeSummarizer()
    results = {}
    for strat in ("truncate", "summarize"):
        c2 = copy.deepcopy(conv)
        cm = ContextManager(cfg(strategy=strat), client)
        before, budget = cm.usage(c2)
        cm.ensure_fits(c2)
        after, _ = cm.usage(c2)
        results[strat] = (before, after)
        assert after < before
        assert after <= budget

    # both start from the same place
    assert results["truncate"][0] == results["summarize"][0]


def test_compare_strategies_prints_a_table():
    out = compare_strategies(big_conversation(), cfg(), FakeSummarizer())
    assert "truncate" in out and "summarize" in out
    assert "before" in out and "saved" in out


def test_what_truncation_loses_that_summarization_keeps():
    """The concrete version of the trade-off, not the hand-wave.

    A fact stated early and never repeated: truncation destroys it,
    summarization has a chance of carrying it forward. That 'chance' is the
    honest word -- the summary is model output, and if it writes 'discussed
    the config file' the filename is just as gone.
    """
    conv = Conversation("sys")
    conv.add_user("The API key lives in config/prod.yaml under the name UPSTREAM_TOKEN.")
    conv.add_assistant("Got it.")
    for i in range(20):
        conv.add_user(f"unrelated question {i} " + ("padding " * 60))
        conv.add_assistant(f"unrelated answer {i} " + ("padding " * 60))

    trunc = copy.deepcopy(conv)
    ContextManager(cfg(strategy="truncate")).ensure_fits(trunc)
    assert not any("UPSTREAM_TOKEN" in m.content for m in trunc.messages)

    summ = copy.deepcopy(conv)
    client = FakeSummarizer("User said the API key is in config/prod.yaml as UPSTREAM_TOKEN.")
    ContextManager(cfg(strategy="summarize"), client).ensure_fits(summ)
    assert any("UPSTREAM_TOKEN" in m.content for m in summ.messages)


# --- tool result trimming ---------------------------------------------


def test_small_results_pass_through_untouched():
    cm = ContextManager(cfg())
    r = ToolResult(call_id="c1", name="grep", content="three lines\nof\noutput")
    out = cm.trim_tool_result(r)
    assert out.content == "three lines\nof\noutput"
    assert not out.truncated


def test_huge_result_is_trimmed_and_spilled(tmp_path):
    c = cfg(max_tool_result_chars=500)
    c.root = tmp_path
    cm = ContextManager(c)
    body = "\n".join(f"src/file{i}.py:{i}: match" for i in range(500))
    r = ToolResult(call_id="c1", name="grep", content=body)

    out = cm.trim_tool_result(r)
    assert out.truncated
    assert len(out.content) < len(body)
    assert "characters trimmed" in out.content
    # full thing is recoverable, not gone
    assert (tmp_path / ".spill" / "grep_c1.txt").read_text() == body
    assert out.spill_path.startswith(".spill/")


def test_trim_keeps_head_and_tail_not_just_head(tmp_path):
    # structured output puts the summary line at the END. head-only
    # truncation reliably cuts off the part that mattered.
    c = cfg(max_tool_result_chars=400)
    c.root = tmp_path
    cm = ContextManager(c)
    body = "FIRST LINE\n" + ("middle padding line\n" * 200) + "LAST LINE: 3 errors"
    out = cm.trim_tool_result(ToolResult(call_id="c2", name="bash", content=body))
    assert "FIRST LINE" in out.content
    assert "LAST LINE: 3 errors" in out.content


def test_spill_path_is_readable_by_the_agent(tmp_path, monkeypatch):
    # the marker names a file; that file has to actually be reachable by
    # read_file or the "read it if you need the middle" line is a lie
    from secret_agent.tools.fs import ReadFile

    monkeypatch.setenv("SA_ROOT", str(tmp_path))
    c = cfg(max_tool_result_chars=300)
    c.root = tmp_path
    cm = ContextManager(c)
    body = "x" * 5000
    out = cm.trim_tool_result(ToolResult(call_id="c3", name="grep", content=body))
    got = ReadFile().run(path=out.spill_path)
    assert "xxx" in got


# --- reporting --------------------------------------------------------


def test_report_is_empty_when_nothing_happened():
    assert "no compaction needed" in ContextManager(cfg()).report()


def test_report_shows_the_before_and_after():
    cm = ContextManager(cfg(strategy="truncate"))
    cm.ensure_fits(big_conversation())
    rep = cm.report()
    assert "compacted (truncate)" in rep
    assert "->" in rep and "tokens" in rep


def test_compaction_str_is_readable():
    c = Compaction(fired=True, strategy="truncate", before=1000, after=400,
                   dropped_turns=5, cost_ms=1.2)
    s = str(c)
    assert "1000 -> 400" in s and "-600" in s and "60%" in s
