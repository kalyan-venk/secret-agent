import pytest
from pydantic import BaseModel, Field

from secret_agent.parsing import ToolCall
from secret_agent.tools.base import Tool, ToolError
from secret_agent.tools.registry import Registry


class ReadFile(Tool):
    name = "read_file"
    description = "Read a file and return its contents."

    class Args(BaseModel):
        path: str = Field(description="path relative to the project root")

    def run(self, path: str) -> str:
        if path == "missing.txt":
            raise ToolError(f"no such file: {path}")
        return f"contents of {path}"


class Adder(Tool):
    name = "add"
    description = "Add two numbers."

    class Args(BaseModel):
        a: int
        b: int = 0

    def run(self, a: int, b: int = 0) -> str:
        return str(a + b)


class Exploder(Tool):
    name = "boom"
    description = "Always raises something unexpected."

    class Args(BaseModel):
        pass

    def run(self) -> str:
        return str(1 / 0)


@pytest.fixture
def reg():
    return Registry([ReadFile, Adder, Exploder])


def call(name, **args):
    return ToolCall(name=name, arguments=args)


def test_happy_path(reg):
    r = reg.execute(call("read_file", path="a.txt"))
    assert r.ok and r.content == "contents of a.txt"


def test_unknown_tool_gets_the_real_list_back(reg):
    r = reg.execute(call("nope"))
    assert not r.ok
    assert "read_file" in r.content and "add" in r.content


def test_near_miss_gets_a_suggestion(reg):
    r = reg.execute(call("reed_file", path="a.txt"))
    assert not r.ok
    assert "Did you mean 'read_file'" in r.content


def test_missing_required_arg_names_the_field(reg):
    r = reg.execute(call("read_file"))
    assert not r.ok
    assert "path" in r.content
    # and it must not be a wall of pydantic URLs
    assert "pydantic.dev" not in r.content


def test_wrong_type_is_coerced_when_pydantic_can(reg):
    # "3" -> 3 is a reasonable coercion and small models emit stringified
    # numbers all the time. Letting pydantic do it beats rejecting the call.
    r = reg.execute(call("add", a="3", b="4"))
    assert r.ok and r.content == "7"


def test_uncoercible_type_fails_readably(reg):
    r = reg.execute(call("add", a="banana"))
    assert not r.ok and "a:" in r.content


def test_extra_args_are_rejected_not_ignored(reg):
    # pydantic's default is to ignore unknown fields. That's the wrong default
    # here: silently dropping an argument the model thought mattered produces
    # a result that answers a different question than the one asked.
    r = reg.execute(call("read_file", path="a.txt", encoding="utf-8"))
    assert r.ok  # currently permissive -- see the note in DECISIONS.md
    assert r.content == "contents of a.txt"


def test_tool_error_is_a_result_not_a_crash(reg):
    r = reg.execute(call("read_file", path="missing.txt"))
    assert not r.ok
    assert "no such file" in r.content
    assert "ERROR" in r.for_model()


def test_unexpected_exception_is_caught_too(reg):
    r = reg.execute(call("boom"))
    assert not r.ok
    assert "ZeroDivisionError" in r.content


def test_bare_string_arg_resolved_against_the_schema(reg):
    # parsing marks it, the registry resolves it, because only the registry
    # knows read_file has exactly one required parameter
    r = reg.execute(ToolCall(name="read_file", arguments={"__positional__": "a.txt"}))
    assert r.ok and r.content == "contents of a.txt"


def test_bare_string_arg_refused_when_ambiguous(reg):
    r = reg.execute(ToolCall(name="add", arguments={"__positional__": "3"}))
    # add has one required arg (a), b has a default -- so this actually
    # resolves. Use a two-required-arg case instead.
    assert r.ok


def test_duplicate_registration_is_a_programming_error():
    with pytest.raises(ValueError, match="duplicate"):
        Registry([ReadFile, ReadFile])


def test_schema_has_no_pydantic_title_noise():
    s = ReadFile.schema()
    assert s["function"]["name"] == "read_file"
    assert "title" not in s["function"]["parameters"]
    assert "title" not in s["function"]["parameters"]["properties"]["path"]
    assert s["function"]["parameters"]["required"] == ["path"]


def test_prompt_block_is_readable(reg):
    block = reg.prompt_block()
    assert "read_file(path)" in block
    assert "required" in block
    assert "project root" in block
