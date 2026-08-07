import json

from secret_agent.conversation import Conversation, Message


def test_system_stays_first_and_is_replaced_not_appended():
    c = Conversation("you are a cat")
    c.add_user("hi")
    c.add_system("you are a dog")
    assert c.messages[0].role == "system"
    assert c.messages[0].content == "you are a dog"
    assert sum(1 for m in c.messages if m.role == "system") == 1


def test_wire_shape_drops_bookkeeping_fields():
    c = Conversation("sys")
    c.add_user("q")
    c.add_assistant("a")
    c.add_tool_result("call_1", "read_file", "contents")
    wire = c.to_wire()
    assert wire[0] == {"role": "system", "content": "sys"}
    assert wire[3] == {"role": "tool", "content": "contents", "name": "read_file"}
    for w in wire:
        assert "pinned" not in w and "created_at" not in w


def test_turns_group_tool_results_with_the_user_message_that_caused_them():
    c = Conversation("sys")
    c.add_user("do a thing")
    c.add_assistant("calling tool")
    c.add_tool_result("c1", "grep", "found")
    c.add_assistant("done")
    c.add_user("another")
    c.add_assistant("ok")

    turns = c.turns()
    assert len(turns) == 2
    assert len(turns[0]) == 4
    assert [m.role for m in turns[0]] == ["user", "assistant", "tool", "assistant"]
    assert len(turns[1]) == 2


def test_three_turn_conversation_still_carries_turn_one(tmp_path):
    # the phase 1 "done when". Not a model test -- it's a payload test. The
    # point is that turn 3's request literally contains turn 1's text.
    c = Conversation("sys")
    c.add_user("my name is Ada")
    c.add_assistant("noted")
    c.add_user("I live in Chicago")
    c.add_assistant("noted")
    c.add_user("what is my name?")

    payload = json.dumps(c.to_wire())
    assert "Ada" in payload
    assert "Chicago" in payload
    assert len(c.to_wire()) == 6


def test_roundtrip_through_disk(tmp_path):
    c = Conversation("sys")
    c.add_user("hello")
    c.add_tool_result("c1", "grep", "x")
    p = tmp_path / "conv.json"
    c.save(p)

    back = Conversation.load(p)
    assert [m.role for m in back.messages] == ["system", "user", "tool"]
    assert back.messages[2].name == "grep"
    assert back.messages[0].pinned is True


def test_transcript_truncates_long_bodies():
    c = Conversation()
    c.add_user("x" * 2000)
    assert "...[cut]" in c.transcript()


# --- hosted wire format ------------------------------------------------
#
# The bug these two tests pin: Groq (and any real OpenAI-compatible
# provider) rejects a role=tool message that doesn't carry a tool_call_id
# matching an id inside a "tool_calls" array on the assistant message right
# before it. The 21 pre-existing tests in test_llm.py never caught this
# because they build messages as raw dicts by hand and never serialize a
# real Conversation -- this is the test that closes that gap. Against the
# pre-fix to_wire() (no provider argument, no tool_calls field ever emitted,
# tool messages never carrying tool_call_id) this fails immediately: calling
# to_wire(provider=...) raises TypeError, and even ignoring that, the tool
# message it produced had no "tool_call_id" key at all.


def test_hosted_wire_format_pairs_tool_call_id_with_assistant_tool_calls():
    c = Conversation("sys")
    c.add_user("what error does Meridian return under legal hold?")
    c.add_assistant(
        '{"name": "search_docs", "arguments": {"query": "legal hold"}}',
        tool_calls=[
            {"id": "call_abc123", "name": "search_docs", "arguments": {"query": "legal hold"}}
        ],
    )
    c.add_tool_result("call_abc123", "search_docs", "error code LH-409, not retryable")

    wire = c.to_wire(provider="hosted")
    assistant_msg = wire[2]
    tool_msg = wire[3]

    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"] == [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {"name": "search_docs", "arguments": '{"query": "legal hold"}'},
        }
    ]

    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_abc123"
    # the id genuinely pairs up, not just "both non-empty"
    assert tool_msg["tool_call_id"] == assistant_msg["tool_calls"][0]["id"]


def test_ollama_wire_format_is_unchanged_by_the_hosted_addition():
    """Same conversation, provider="ollama" (also the default with no
    argument at all): must be byte-for-byte what to_wire() always produced --
    no tool_call_id, no tool_calls array. Ollama's own tolerance for those
    extra keys has not been verified, so the safe default is to not send them.
    """
    c = Conversation("sys")
    c.add_user("what error does Meridian return under legal hold?")
    c.add_assistant(
        '{"name": "search_docs", "arguments": {"query": "legal hold"}}',
        tool_calls=[
            {"id": "call_abc123", "name": "search_docs", "arguments": {"query": "legal hold"}}
        ],
    )
    c.add_tool_result("call_abc123", "search_docs", "error code LH-409, not retryable")

    wire_default = c.to_wire()
    wire_explicit = c.to_wire(provider="ollama")
    assert wire_default == wire_explicit

    assistant_msg = wire_default[2]
    tool_msg = wire_default[3]
    assert "tool_calls" not in assistant_msg
    assert "tool_call_id" not in tool_msg
    assert tool_msg == {"role": "tool", "content": "error code LH-409, not retryable", "name": "search_docs"}
