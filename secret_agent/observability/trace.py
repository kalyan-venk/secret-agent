"""Per-step tracing for an agent run.

You cannot debug a multi-step agent run by staring at the final answer. When a
run takes 40 seconds or loops five times or calls the wrong tool, the question
is always "which step, and was it the model or the tool", and the loop threw all
of that away once it returned. This records one structured line per step while
the run happens, then a summary line at the end.

It hangs off the loop's existing `on_step` callback (agent.py already calls it
after every step), so nothing in the loop changes to turn tracing on. Each line
is JSON so it greps and loads without a parser:

    {"run_id": "...", "step": 1, "model_ms": 812.4, "step_ms": 831.0,
     "prompt_tokens": 1204, "completion_tokens": 37, "compacted": false,
     "repaired": false,
     "tools": [{"name": "search_docs", "duration_ms": 12.1, "ok": true,
                "error": null}]}

`model_ms` is the model call alone; `step_ms` is the whole step including the
tool runs, so a slow step is immediately attributable to one or the other.
Per-tool `duration_ms` comes straight off ToolResult, which the registry has
always measured. Tokens are whatever the provider reported (Ollama and Groq both
do; a test double may not, in which case they are 0).

Writing is best-effort. A trace that fails to write must never be the reason an
agent run dies, same rule as the LLM call log and the context spill.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from ..agent import Step


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


class Tracer:
    """Collects per-step records and writes them as JSON lines.

    Pass `tracer.on_step` as the Agent's `on_step`. Call `dump_summary()` after
    the run (or in a finally, so a failed run still gets its summary).
    """

    def __init__(self, path: str | Path | None = None, stream: TextIO | None = None,
                 run_id: str | None = None):
        self.run_id = run_id or _new_run_id()
        self.records: list[dict] = []
        self._path = Path(path) if path else None
        # Default to stderr so trace lines never mix into the agent's answer on
        # stdout. An explicit stream (or a file path) overrides that.
        self._stream = stream if stream is not None else (None if self._path else sys.stderr)
        self._started = time.time()

    # --- the callback the loop drives -------------------------------------

    def on_step(self, step: "Step") -> None:
        rec = self._record(step)
        self.records.append(rec)
        self._write(rec)

    def _record(self, step: "Step") -> dict:
        tools = [
            {
                "name": r.name,
                "duration_ms": r.duration_ms,
                "ok": r.ok,
                # only the failure text, capped -- a 4KB stack trace does not
                # belong on a trace line
                "error": None if r.ok else (r.content or "")[:200],
            }
            for r in step.results
        ]
        return {
            "run_id": self.run_id,
            "step": step.n,
            "model_ms": round(step.model_ms, 1),
            "step_ms": round(step.elapsed_s * 1000, 1),
            "prompt_tokens": step.prompt_tokens,
            "completion_tokens": step.completion_tokens,
            "compacted": step.compacted,
            "repaired": step.repaired,
            "tools": tools,
        }

    # --- the end-of-run rollup --------------------------------------------

    def summary(self) -> dict:
        """One dict of aggregates over the whole run.

        Per-tool latency is split out because "the run was slow" is never the
        useful sentence -- "search_docs averaged 1.2s over 3 calls" is.
        """
        steps = self.records
        tool_stats: dict[str, dict] = {}
        failures = 0
        for rec in steps:
            for t in rec["tools"]:
                s = tool_stats.setdefault(t["name"], {"calls": 0, "total_ms": 0.0, "failures": 0})
                s["calls"] += 1
                s["total_ms"] = round(s["total_ms"] + t["duration_ms"], 2)
                if not t["ok"]:
                    s["failures"] += 1
                    failures += 1
        for s in tool_stats.values():
            s["avg_ms"] = round(s["total_ms"] / s["calls"], 2) if s["calls"] else 0.0

        return {
            "run_id": self.run_id,
            "kind": "summary",
            "steps": len(steps),
            "tool_calls": sum(len(r["tools"]) for r in steps),
            "tool_failures": failures,
            "repaired_steps": sum(1 for r in steps if r["repaired"]),
            "compactions": sum(1 for r in steps if r["compacted"]),
            "prompt_tokens": sum(r["prompt_tokens"] for r in steps),
            "completion_tokens": sum(r["completion_tokens"] for r in steps),
            "model_ms": round(sum(r["model_ms"] for r in steps), 1),
            "wall_ms": round((time.time() - self._started) * 1000, 1),
            "tools": tool_stats,
        }

    def dump_summary(self) -> dict:
        s = self.summary()
        self._write(s)
        return s

    def summary_line(self) -> str:
        """A one-line human version for a terminal, not for a log file."""
        s = self.summary()
        parts = [f"{s['steps']} steps", f"{s['tool_calls']} tool calls"]
        if s["tool_failures"]:
            parts.append(f"{s['tool_failures']} failed")
        if s["repaired_steps"]:
            parts.append(f"{s['repaired_steps']} repaired")
        if s["compactions"]:
            parts.append(f"{s['compactions']} compactions")
        parts.append(f"{s['prompt_tokens']}+{s['completion_tokens']} tokens")
        parts.append(f"model {s['model_ms'] / 1000:.1f}s of {s['wall_ms'] / 1000:.1f}s wall")
        return "trace " + s["run_id"] + ": " + ", ".join(parts)

    # --- writing ----------------------------------------------------------

    def _write(self, rec: dict) -> None:
        line = json.dumps(rec)
        try:
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            if self._stream is not None:
                self._stream.write(line + "\n")
                self._stream.flush()
        except OSError:
            pass
