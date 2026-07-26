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
    c.add_user("my name is Kalyan")
    c.add_assistant("noted")
    c.add_user("I live in Chicago")
    c.add_assistant("noted")
    c.add_user("what is my name?")

    payload = json.dumps(c.to_wire())
    assert "Kalyan" in payload
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
