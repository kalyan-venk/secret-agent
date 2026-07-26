import pytest

from secret_agent.parsing import (
    STATS,
    balanced_spans,
    loads_forgiving,
    parse_tool_calls,
    strip_fences,
)

from .fixtures.malformed import CASES, JSON_THE_MODEL_IS_TALKING_ABOUT, EXTRA_ARGS

KNOWN = {"read_file", "write_file", "grep", "list_dir", "bash"}


@pytest.mark.parametrize("label,text,expected", CASES, ids=[c[0] for c in CASES])
def test_malformed_fixture(label, text, expected):
    res = parse_tool_calls(text, known_names=KNOWN)
    got = [(c.name, c.arguments) for c in res.calls]
    assert got == expected, f"{label}: {got!r}"


def test_bare_string_arg_is_flagged_for_the_registry():
    # parsing can't know the parameter name; it marks it and the registry,
    # which has the schema, resolves it. See test_registry.
    res = parse_tool_calls(
        '{"name": "read_file", "arguments": "notes.txt"}', known_names=KNOWN
    )
    assert res.calls[0].arguments == {"__positional__": "notes.txt"}
    assert "args_bare_string" in res.calls[0].repairs


def test_json_the_model_is_merely_discussing_is_not_a_call():
    res = parse_tool_calls(JSON_THE_MODEL_IS_TALKING_ABOUT, known_names=KNOWN)
    assert res.calls == []


def test_extra_args_survive_parsing_and_die_at_validation():
    # deliberate split: the parser's job is JSON, not legality
    res = parse_tool_calls(EXTRA_ARGS, known_names=KNOWN)
    assert res.calls[0].arguments["encoding"] == "utf-8"


def test_prose_is_preserved_separately_from_the_call():
    text = 'Let me check that.\n{"name": "grep", "arguments": {"pattern": "x"}}\nOne sec.'
    res = parse_tool_calls(text, known_names=KNOWN)
    assert len(res.calls) == 1
    assert "Let me check that." in res.text
    assert '"name"' not in res.text


def test_repairs_are_recorded_not_silent():
    # the whole point of the module. A repaired call must be distinguishable
    # from a clean one after the fact.
    clean = parse_tool_calls(
        '{"name": "grep", "arguments": {"pattern": "x"}}', known_names=KNOWN
    )
    dirty = parse_tool_calls(
        "```json\n{'name': 'grep', 'arguments': {'pattern': 'x'},}\n```",
        known_names=KNOWN,
    )
    assert clean.calls[0].was_clean
    assert not dirty.calls[0].was_clean
    assert "fenced" in dirty.calls[0].repairs


def test_stats_counts_repair_rate():
    STATS.reset()
    parse_tool_calls('{"name": "grep", "arguments": {"pattern": "x"}}', known_names=KNOWN)
    parse_tool_calls('```json\n{"name": "grep", "arguments": {"pattern": "x"}}\n```',
                     known_names=KNOWN)
    parse_tool_calls("just talking", known_names=KNOWN)
    assert STATS.completions == 3
    assert STATS.completions_with_calls == 2
    assert STATS.completions_needing_repair == 1
    assert "50.0%" in STATS.summary()
    STATS.reset()


# --- the pieces, tested on their own ----------------------------------


def test_balanced_spans_ignores_braces_inside_strings():
    s = '{"a": "} not the end {"} tail'
    (start, end), = balanced_spans(s)
    assert s[start:end] == '{"a": "} not the end {"}'


def test_balanced_spans_survives_apostrophes_in_prose():
    # regression: "I'll" used to open a string that never closed, and every
    # tool call after it in the completion vanished
    s = """I'll check that for you.
{"name": "grep", "arguments": {"pattern": "x"}}"""
    assert len(balanced_spans(s)) == 1


def test_balanced_spans_handles_nesting():
    s = 'x {"a": {"b": {"c": 1}}} y'
    (start, end), = balanced_spans(s)
    assert s[start:end] == '{"a": {"b": {"c": 1}}}'


def test_strip_fences_reports_whether_it_did_anything():
    out, hit = strip_fences("no fences here")
    assert not hit and out == "no fences here"
    out, hit = strip_fences("```json\n{}\n```")
    assert hit and out.strip() == "{}"


def test_forgiving_loader_ladder():
    assert loads_forgiving('{"a": 1}') == ({"a": 1}, [])

    obj, r = loads_forgiving('{"a": 1,}')
    assert obj == {"a": 1} and r == ["trailing_comma"]

    obj, r = loads_forgiving("{'a': 1}")
    assert obj == {"a": 1} and "python_literal" in r

    obj, r = loads_forgiving('{"a": 1')
    assert obj == {"a": 1} and "unclosed_braces" in r

    obj, r = loads_forgiving("this is not json at all")
    assert obj is None


def test_forgiving_loader_gives_up_rather_than_guessing():
    # there is no repair for this and there shouldn't be -- inventing one
    # means executing a tool based on a guess about intent
    obj, _ = loads_forgiving('{"name": read_file, "arguments" path}')
    assert obj is None


def test_prose_is_split_around_the_calls_not_merged():
    # the two halves mean different things: before = reasoning,
    # after = narration of a result that doesn't exist yet
    text = ('Let me check.\n{"name": "grep", "arguments": {"pattern": "x"}}\n'
            'It returned three matches.')
    res = parse_tool_calls(text, known_names=KNOWN)
    assert res.prose_before == "Let me check."
    assert res.prose_after == "It returned three matches."


def test_prose_split_is_empty_when_there_are_no_calls():
    res = parse_tool_calls("just talking", known_names=KNOWN)
    assert res.prose_before == "" and res.prose_after == ""


def test_prose_split_spans_from_first_call_to_last():
    text = ('before\n{"name": "grep", "arguments": {"pattern": "a"}}\n'
            'middle\n{"name": "grep", "arguments": {"pattern": "b"}}\nafter')
    res = parse_tool_calls(text, known_names=KNOWN)
    assert len(res.calls) == 2
    assert res.prose_before == "before"
    assert res.prose_after == "after"
    assert "middle" not in res.prose_before and "middle" not in res.prose_after
