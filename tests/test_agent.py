"""Loop tests, all offline.

The client is a scripted stub, so these test MY control flow rather than
llama3.1's mood on a given afternoon. There's one live test at the bottom
that actually talks to ollama; it's marked and excluded by default.
"""

import pytest
from pydantic import BaseModel

from secret_agent.agent import Agent, AgentLoopExhausted, ParseRetriesExhausted
from secret_agent.config import Config
from secret_agent.llm import Completion, Usage
from secret_agent.tools.base import Tool, ToolError
from secret_agent.tools.registry import Registry


class Scripted:
    """Hands back canned completions in order and records what it was sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.payloads = []

    def complete(self, messages, tools=None):
        self.payloads.append(messages)
        text = self.responses.pop(0) if self.responses else "done, nothing left to say"
        return Completion(text=text, usage=Usage(prompt_tokens=len(str(messages)) // 4))

    def count_tokens(self, text):
        return len(text) // 4


class ScriptedHosted(Scripted):
    """Same as Scripted, but declares WIRE_FORMAT="hosted" like the real
    HostedClient does. Used to prove agent.py actually threads the client's
    declared wire format through to Conversation.to_wire() -- see
    test_hosted_tool_turn_produces_a_paired_wire_message below. A plain
    Scripted (no WIRE_FORMAT attribute) exercises the getattr(...,
    "ollama") fallback, which the rest of this file's tests already cover
    implicitly since none of them broke when that fallback was added.
    """

    WIRE_FORMAT = "hosted"


# --- tools -----------------------------------------------------------

CALLS = []


class Echo(Tool):
    name = "echo"
    description = "Echo a string back."

    class Args(BaseModel):
        text: str

    def run(self, text: str) -> str:
        CALLS.append(("echo", text))
        return text


class Counter(Tool):
    name = "counter"
    description = "Return the number of times it has been called."

    class Args(BaseModel):
        pass

    def run(self) -> str:
        CALLS.append(("counter", None))
        return str(len([c for c in CALLS if c[0] == "counter"]))


class Flaky(Tool):
    name = "flaky"
    description = "Raises."

    class Args(BaseModel):
        pass

    def run(self) -> str:
        raise ToolError("upstream is on fire")


@pytest.fixture(autouse=True)
def _clear():
    CALLS.clear()


@pytest.fixture
def cfg():
    return Config(max_iterations=6, tool_mode="prompted")


@pytest.fixture
def reg():
    return Registry([Echo, Counter, Flaky])


def call(name, **args):
    import json
    return json.dumps({"name": name, "arguments": args})


# --- the basics ------------------------------------------------------


def test_no_tool_call_means_done(reg, cfg):
    a = Agent(reg, Scripted("The answer is 42."), cfg)
    run = a.run("what is the answer")
    assert run.answer == "The answer is 42."
    assert run.iterations == 1
    assert run.tool_calls == 0


def test_one_tool_then_answer(reg, cfg):
    a = Agent(reg, Scripted(call("echo", text="hi"), "It said hi."), cfg)
    run = a.run("use echo")
    assert run.answer == "It said hi."
    assert CALLS == [("echo", "hi")]
    assert run.iterations == 2


def test_hosted_tool_turn_produces_a_paired_wire_message(reg, cfg):
    """The BLOCKING 1 regression: a hosted-shaped client (WIRE_FORMAT=
    "hosted", same as the real HostedClient/Groq) doing a tool-using run in
    the default "prompted" tool_mode -- exactly what scripts/hosted_eval.py
    exercises. Before the fix, agent.py always called
    self.conversation.to_wire() with no provider argument, so every client
    got the Ollama shape regardless of what it actually needed, and the tool
    result message never carried tool_call_id. Groq 400s on that.

    This checks the SECOND request's payload (the one built after the tool
    call + result are already in history) has the assistant message's
    tool_calls[].id matching the following tool message's tool_call_id --
    the exact pairing Groq/OpenAI require.
    """
    client = ScriptedHosted(call("echo", text="hi"), "It said hi.")
    a = Agent(reg, client, cfg)
    run = a.run("use echo")
    assert run.answer == "It said hi."

    second_payload = client.payloads[1]
    assistant_msg = next(m for m in second_payload if m["role"] == "assistant" and m.get("tool_calls"))
    tool_msg = next(m for m in second_payload if m["role"] == "tool")

    assert "tool_calls" in assistant_msg
    call_id = assistant_msg["tool_calls"][0]["id"]
    assert tool_msg["tool_call_id"] == call_id
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "echo"


def test_three_chained_calls_complete_end_to_end(reg, cfg):
    # the phase 3 "done when"
    a = Agent(
        reg,
        Scripted(
            call("counter"),
            call("counter"),
            call("counter"),
            "I called it three times.",
        ),
        cfg,
    )
    run = a.run("call counter three times")
    assert run.tool_calls == 3
    assert run.iterations == 4
    assert run.answer == "I called it three times."
    # and the results genuinely accumulated in history
    tool_msgs = [m for m in run.conversation.messages if m.role == "tool"]
    assert [m.content for m in tool_msgs] == ["1", "2", "3"]


def test_prose_alongside_a_call_does_not_stop_the_loop(reg, cfg):
    # this is the one that bit me. "Sure, let me check!" + a call is not a
    # final answer, but a naive "did it produce text" check says it is.
    a = Agent(
        reg,
        Scripted(
            'Sure, let me echo that for you.\n' + call("echo", text="hi"),
            "Done.",
        ),
        cfg,
    )
    run = a.run("echo hi")
    assert run.answer == "Done."
    assert CALLS == [("echo", "hi")]


def test_parallel_calls_return_in_one_turn(reg, cfg):
    two = '[{"name": "echo", "arguments": {"text": "a"}}, ' \
          '{"name": "echo", "arguments": {"text": "b"}}]'
    a = Agent(reg, Scripted(two, "both done"), cfg)
    run = a.run("echo a and b")
    assert CALLS == [("echo", "a"), ("echo", "b")]
    assert run.iterations == 2  # NOT 3 -- both results went back together
    assert len(run.steps[0].results) == 2


# --- the four failure modes -----------------------------------------


def test_runaway_hits_the_cap_and_raises(reg):
    c = Config(max_iterations=4)
    a = Agent(reg, Scripted(*[call("counter")] * 20), c)
    with pytest.raises(AgentLoopExhausted) as e:
        a.run("loop forever")
    assert "4 iterations" in str(e.value)
    # the partial run is attached, not thrown away
    assert len(e.value.steps) == 4
    assert e.value.conversation is not None


def test_tool_exception_becomes_a_message_the_model_can_read(reg, cfg):
    a = Agent(reg, Scripted(call("flaky"), "I see, it failed."), cfg)
    run = a.run("use flaky")
    tool_msg = [m for m in run.conversation.messages if m.role == "tool"][0]
    assert "upstream is on fire" in tool_msg.content
    assert tool_msg.content.startswith("ERROR")
    assert run.answer == "I see, it failed."


def test_hallucinated_tool_name_comes_back_as_a_correctable_error(reg, cfg):
    a = Agent(reg, Scripted(call("ecko", text="hi"), call("echo", text="hi"), "ok"), cfg)
    run = a.run("x")
    first = [m for m in run.conversation.messages if m.role == "tool"][0]
    assert "no tool named 'ecko'" in first.content
    assert "Did you mean 'echo'" in first.content
    assert CALLS == [("echo", "hi")]  # it recovered


def test_far_off_tool_name_gets_no_suggestion(reg, cfg):
    # difflib cutoff is 0.7 and stays there. A confidently wrong suggestion
    # is worse than none, because the model will take it.
    a = Agent(reg, Scripted(call("frobnicate", text="hi"), "ok"), cfg)
    run = a.run("x")
    first = [m for m in run.conversation.messages if m.role == "tool"][0]
    assert "Did you mean" not in first.content
    assert "counter, echo, flaky" in first.content


def test_unparseable_output_is_retried_then_raises(reg):
    c = Config(max_iterations=8, max_parse_retries=2)
    junk = '{"name": echo, "arguments" broken}'
    a = Agent(reg, Scripted(junk, junk, junk, "fine, plain answer"), c)
    with pytest.raises(ParseRetriesExhausted) as e:
        a.run("x")
    # it must NOT hand the broken json back as an answer
    assert "3 times in a row" in str(e.value)
    nudges = [m for m in e.value.conversation.messages
              if m.role == "user" and "could not be parsed" in m.content]
    assert len(nudges) == 2


def test_model_that_recovers_after_a_nudge_is_fine(reg):
    c = Config(max_iterations=8, max_parse_retries=2)
    junk = '{"name": echo, "arguments" broken}'
    a = Agent(reg, Scripted(junk, call("echo", text="ok"), "answer"), c)
    run = a.run("x")
    assert run.answer == "answer"
    assert CALLS == [("echo", "ok")]


def test_parse_failure_counter_resets_after_a_good_call(reg):
    c = Config(max_iterations=8, max_parse_retries=2)
    junk = '{"name": echo, "arguments" broken}'
    a = Agent(reg, Scripted(junk, call("echo", text="ok"), junk, junk, "answer"), c)
    run = a.run("x")
    # if the counter didn't reset, the 2nd junk would be failure #3 and this
    # would raise instead of finishing
    assert run.answer == "answer"


# --- bookkeeping ------------------------------------------------------


def test_repair_rate_is_reported(reg, cfg):
    dirty = "```json\n" + call("echo", text="hi") + "\n```"
    a = Agent(reg, Scripted(dirty, "done"), cfg)
    run = a.run("x")
    assert run.repair_rate == 1.0
    assert "repair rate 100%" in run.summary()


def test_history_is_what_the_model_actually_sees(reg, cfg):
    client = Scripted(call("echo", text="hi"), "done")
    a = Agent(reg, client, cfg)
    a.run("please echo hi")

    # payload on iteration 2 -- reconstruct exactly what went over the wire
    second = client.payloads[1]
    roles = [m["role"] for m in second]
    assert roles == ["system", "user", "assistant", "tool"]
    assert "echo(text)" in second[0]["content"]  # tool schemas live in system
    assert second[1]["content"] == "please echo hi"
    assert second[3]["content"] == "hi"


def test_native_mode_skips_the_schema_block_in_the_prompt(reg):
    c = Config(tool_mode="native")
    a = Agent(reg, Scripted("done"), c)
    assert "echo(text)" not in a.conversation.system_prompt


# --- live -------------------------------------------------------------


@pytest.mark.live
def test_against_real_ollama(reg):
    """Actually run the loop against llama3.1:8b.

    Asserts on MY control flow, not on the model's prose. First version of
    this test checked that the final answer contained "pineapple" and it
    failed -- not because the loop broke but because llama3.1 called echo
    correctly, got the right result, and then wandered off and called an
    unrelated tool before answering about that instead.

    That is the model being 8B. Asserting on it makes the suite a coin flip.
    So: assert the call happened, the args were right, and the real result
    reached history. Those are mine to get right.
    """
    from secret_agent.llm import OllamaClient

    c = Config(max_iterations=6)
    a = Agent(reg, OllamaClient(c), c)
    run = a.run("Call the echo tool with the text 'pineapple', then tell me what it returned.")

    assert ("echo", "pineapple") in CALLS
    tool_msgs = [m for m in run.conversation.messages if m.role == "tool"]
    assert any(m.name == "echo" and m.content == "pineapple" for m in tool_msgs)
    assert run.iterations <= 6


@pytest.mark.live
def test_model_narrates_a_result_it_has_not_got_yet(reg):
    """Runs the scenario behind Agent.record_assistant_turn so the transcript
    is easy to go and look at. Nothing is asserted about the narration itself
    -- whether the model fabricates on any given run is up to the model, and a
    test that depends on that is a coin flip.
    """
    from secret_agent.llm import OllamaClient

    c = Config(max_iterations=4)
    a = Agent(reg, OllamaClient(c), c)
    try:
        run = a.run("Call echo with 'pineapple' and tell me what it returned.")
    except Exception:
        pytest.skip("model didn't cooperate; this test is documentation")
    assert run.conversation.transcript()


# --- what goes into history when the model talks AND calls ------------


def test_reasoning_before_a_call_is_kept(reg, cfg):
    a = Agent(reg, Scripted(
        "I need to check that first.\n" + call("echo", text="hi"), "done"), cfg)
    run = a.run("x")
    msg = [m for m in run.conversation.messages if m.role == "assistant"][0]
    assert "I need to check that first." in msg.content


def test_narration_after_a_call_is_dropped(reg, cfg):
    # the real llama3.1 failure: it states the result before the tool runs.
    # keeping it puts a fabricated value in history next to the true one.
    a = Agent(reg, Scripted(
        call("echo", text="pineapple") + '\n\nThe echo tool returned: "kiwi"',
        "done"), cfg)
    run = a.run("x")
    msg = [m for m in run.conversation.messages if m.role == "assistant"][0]
    assert "kiwi" not in msg.content
    # and the real result is still there, from the tool
    tool = [m for m in run.conversation.messages if m.role == "tool"][0]
    assert tool.content == "pineapple"


def test_the_models_own_json_is_kept_verbatim(reg, cfg):
    # measured: paraphrasing the call into "[calling echo(text='hi')]" costs
    # ~25% more model calls. It seems to need its own format back.
    # scripts/turn_policy.py
    a = Agent(reg, Scripted(call("echo", text="hi"), "done"), cfg)
    run = a.run("x")
    msg = [m for m in run.conversation.messages if m.role == "assistant"][0]
    assert msg.content == '{"name": "echo", "arguments": {"text": "hi"}}'


def test_fences_are_not_carried_into_history(reg, cfg):
    # call.raw is the extracted JSON, so the fence goes away for free
    fenced = "```json\n" + call("echo", text="hi") + "\n```"
    a = Agent(reg, Scripted(fenced, "done"), cfg)
    run = a.run("x")
    msg = [m for m in run.conversation.messages if m.role == "assistant"][0]
    assert "```" not in msg.content
    assert '"echo"' in msg.content


def test_a_final_answer_with_no_call_is_recorded_verbatim(reg, cfg):
    a = Agent(reg, Scripted("Here is my full answer, unabridged."), cfg)
    run = a.run("x")
    msg = [m for m in run.conversation.messages if m.role == "assistant"][0]
    assert msg.content == "Here is my full answer, unabridged."


def test_both_calls_are_recorded_for_a_parallel_turn(reg, cfg):
    two = '[{"name": "echo", "arguments": {"text": "a"}}, ' \
          '{"name": "echo", "arguments": {"text": "b"}}]'
    a = Agent(reg, Scripted(two, "done"), cfg)
    run = a.run("x")
    msg = [m for m in run.conversation.messages if m.role == "assistant"][0]
    assert '"a"' in msg.content and '"b"' in msg.content


def test_an_array_of_calls_is_not_written_into_history_twice(reg, cfg):
    """Regression, external review 2026-07-25.

    When ONE JSON span parses to an array of N calls, every ToolCall.raw is
    that same whole span -- parsing sets raw to the enclosing text, not the
    individual element. Extending naively put the array into history N times:
    token bloat, plus N duplicate call records for the model to read back.
    """
    two = '[{"name": "echo", "arguments": {"text": "a"}}, ' \
          '{"name": "echo", "arguments": {"text": "b"}}]'
    a = Agent(reg, Scripted(two, "done"), cfg)
    run = a.run("x")
    msg = [m for m in run.conversation.messages if m.role == "assistant"][0]
    assert msg.content.count('"name": "echo"') == 2   # the array, once
    assert msg.content == two
    assert len(run.steps[0].results) == 2             # both still executed


def test_separate_spans_are_both_kept(reg, cfg):
    """The other side of the dedup: two DIFFERENT calls in two separate JSON
    objects must both survive. Deduping on raw must not collapse those."""
    two = ('{"name": "echo", "arguments": {"text": "a"}}\n'
           '{"name": "echo", "arguments": {"text": "b"}}')
    a = Agent(reg, Scripted(two, "done"), cfg)
    run = a.run("x")
    msg = [m for m in run.conversation.messages if m.role == "assistant"][0]
    assert msg.content.count('"name": "echo"') == 2
    assert '"a"' in msg.content and '"b"' in msg.content
