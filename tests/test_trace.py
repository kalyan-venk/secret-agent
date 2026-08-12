"""The observability tracer, offline.

Drives the loop with a scripted client so the trace is deterministic, then reads
the JSON lines back and checks each carries what you need to debug a run: which
step, model vs tool latency, tokens, and the failure text when a tool blows up.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from secret_agent.agent import Agent
from secret_agent.config import Config
from secret_agent.llm import Completion, Usage
from secret_agent.observability import Tracer
from secret_agent.tools.base import Tool, ToolError
from secret_agent.tools.registry import Registry


class Scripted:
    def __init__(self, *responses):
        self.responses = list(responses)

    def complete(self, messages, tools=None):
        text = self.responses.pop(0) if self.responses else "done"
        return Completion(text=text, usage=Usage(prompt_tokens=100, completion_tokens=7))

    def count_tokens(self, text):
        return len(text) // 4


class Echo(Tool):
    name = "echo"
    description = "Echo a string back."

    class Args(BaseModel):
        text: str

    def run(self, text: str) -> str:
        return text


class Boom(Tool):
    name = "boom"
    description = "Always fails."

    class Args(BaseModel):
        pass

    def run(self) -> str:
        raise ToolError("kaboom")


def call(name, **args):
    return json.dumps({"name": name, "arguments": args})


@pytest.fixture
def reg():
    return Registry([Echo, Boom])


@pytest.fixture
def cfg():
    return Config(max_iterations=6, tool_mode="prompted")


def _read_lines(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_one_trace_line_per_step_written_to_file(tmp_path, reg, cfg):
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path=path)
    Agent(reg, Scripted(call("echo", text="hi"), "all done"), cfg,
          on_step=tracer.on_step).run("x")

    lines = _read_lines(path)
    assert len(lines) == 2  # tool step, then the final-answer step
    assert [r["step"] for r in lines] == [1, 2]
    assert all(r["run_id"] == tracer.run_id for r in lines)


def test_a_step_record_has_the_fields_needed_to_debug_it(tmp_path, reg, cfg):
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path=path)
    Agent(reg, Scripted(call("echo", text="hi"), "done"), cfg,
          on_step=tracer.on_step).run("x")

    first = _read_lines(path)[0]
    assert first["prompt_tokens"] == 100
    assert first["completion_tokens"] == 7
    assert "model_ms" in first and "step_ms" in first
    tool = first["tools"][0]
    assert tool["name"] == "echo"
    assert tool["ok"] is True
    assert tool["error"] is None
    assert "duration_ms" in tool


def test_a_tool_failure_is_captured_with_its_error(tmp_path, reg, cfg):
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path=path)
    Agent(reg, Scripted(call("boom"), "gave up"), cfg,
          on_step=tracer.on_step).run("x")

    tool = _read_lines(path)[0]["tools"][0]
    assert tool["name"] == "boom"
    assert tool["ok"] is False
    assert "kaboom" in tool["error"]


def test_the_final_answer_step_has_no_tools(tmp_path, reg, cfg):
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path=path)
    Agent(reg, Scripted("just an answer, no tools"), cfg,
          on_step=tracer.on_step).run("x")

    lines = _read_lines(path)
    assert len(lines) == 1
    assert lines[0]["tools"] == []


def test_summary_aggregates_the_run(reg, cfg):
    tracer = Tracer(stream=None)  # no output, just collect
    Agent(reg, Scripted(call("echo", text="a"), call("boom"), "done"), cfg,
          on_step=tracer.on_step).run("x")

    s = tracer.summary()
    assert s["steps"] == 3
    assert s["tool_calls"] == 2
    assert s["tool_failures"] == 1
    assert s["prompt_tokens"] == 300  # 3 steps * 100
    assert s["tools"]["echo"]["calls"] == 1
    assert s["tools"]["boom"]["failures"] == 1
    assert "avg_ms" in s["tools"]["echo"]


def test_repaired_step_is_flagged_in_the_trace(tmp_path, reg, cfg):
    # a fenced tool call needs repair; the trace must record that so a
    # cross-model comparison isn't silently reading the parser's work as the
    # model's
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path=path)
    dirty = "```json\n" + call("echo", text="hi") + "\n```"
    Agent(reg, Scripted(dirty, "done"), cfg, on_step=tracer.on_step).run("x")

    assert _read_lines(path)[0]["repaired"] is True
