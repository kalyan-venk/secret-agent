"""name -> Tool, plus dispatch.

Also where the permission layer gets consulted, because there is exactly one
place a tool can be invoked from and this is it. If a second call site ever
appears, the guardrails become optional, which is the same as not having them.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

from ..parsing import ToolCall
from .base import Tool, ToolDenied, ToolError, ToolResult, render_tools_for_prompt


class Registry:
    def __init__(self, tools: Iterable[type[Tool]] = (), permissions=None):
        self._tools: dict[str, type[Tool]] = {}
        self.permissions = permissions
        for t in tools:
            self.register(t)

    def register(self, tool: type[Tool]) -> None:
        if not tool.name:
            raise ValueError(f"{tool.__name__} has no name")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> set[str]:
        return set(self._tools)

    def get(self, name: str) -> type[Tool] | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def prompt_block(self) -> str:
        return render_tools_for_prompt(list(self._tools.values()))


    def execute(self, call: ToolCall) -> ToolResult:
        """Run one call. Never raises for a tool-level problem.

        Everything that goes wrong comes back as ToolResult(ok=False) with
        text the model can read and act on. The loop must not crash because
        the model guessed a filename wrong -- that's a normal Tuesday, not an
        exceptional condition.

        Genuine programming errors (a bug in a tool) also get caught here,
        which I went back and forth on. Letting them propagate gives a better
        stack trace; catching them means a half-finished tool doesn't kill a
        long agent run. Catching won, but the traceback is printed when
        verbose so it isn't actually swallowed.
        """
        t0 = time.perf_counter()

        tool = self._tools.get(call.name)
        if tool is None:
            close = _did_you_mean(call.name, self.names)
            hint = f" Did you mean {close!r}?" if close else ""
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content=(
                    f"no tool named {call.name!r}.{hint} "
                    f"Available: {', '.join(sorted(self.names))}"
                ),
                duration_ms=_ms(t0),
            )

        args = dict(call.arguments)

        # parsing couldn't name a bare string arg, but we know the schema, so
        # if there's exactly one required parameter it's unambiguous
        if "__positional__" in args:
            val = args.pop("__positional__")
            req = tool.schema()["function"]["parameters"].get("required", [])
            if len(req) == 1:
                args[req[0]] = val
            else:
                return ToolResult(
                    call_id=call.id,
                    name=call.name,
                    ok=False,
                    content=(
                        f"{call.name} got a bare string but takes {len(req)} required "
                        f"arguments ({', '.join(req)}). Send an object."
                    ),
                    duration_ms=_ms(t0),
                )

        try:
            validated = tool.validate_args(args)
        except ToolError as e:
            return ToolResult(
                call_id=call.id, name=call.name, ok=False,
                content=str(e), duration_ms=_ms(t0),
            )

        # permission check happens AFTER validation so the user is never asked
        # to approve a call that was going to be rejected anyway
        if self.permissions is not None:
            decision = self.permissions.check(tool, validated)
            if not decision.allowed:
                return ToolResult(
                    call_id=call.id, name=call.name, ok=False,
                    content=decision.reason, duration_ms=_ms(t0),
                )

        try:
            out = tool().run(**validated.model_dump())
        except ToolDenied as e:
            return ToolResult(call_id=call.id, name=call.name, ok=False,
                              content=str(e), duration_ms=_ms(t0))
        except ToolError as e:
            return ToolResult(call_id=call.id, name=call.name, ok=False,
                              content=str(e), duration_ms=_ms(t0))
        except Exception as e:  # noqa: BLE001 - see docstring
            import os, traceback
            if os.environ.get("SA_VERBOSE"):
                traceback.print_exc()
            return ToolResult(
                call_id=call.id, name=call.name, ok=False,
                content=f"{type(e).__name__}: {e}", duration_ms=_ms(t0),
            )

        return ToolResult(
            call_id=call.id, name=call.name, ok=True,
            content=out if isinstance(out, str) else str(out),
            duration_ms=_ms(t0),
        )


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def _did_you_mean(name: str, options: set[str]) -> str | None:
    """difflib, but only when it's confident. A bad suggestion is worse than
    none -- the model will take it."""
    import difflib

    m = difflib.get_close_matches(name, list(options), n=1, cutoff=0.7)
    return m[0] if m else None
