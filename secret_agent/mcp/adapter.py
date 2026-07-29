"""Map discovered MCP tools onto the runtime's Tool interface.

The whole point of the extension: an external MCP server's tools become
ordinary tools in the Registry, so every guardrail the native tools already go
through governs them too, with no new code in the loop.

  - parse + repair ladder: Agent._parse already calls parse_tool_calls(text,
    known_names=registry.names). Register an MCP tool and its name flows through
    the identical parser, so a fenced or trailing-comma MCP call is repaired the
    same way a read_file call is.
  - permission layer: Registry.execute calls permissions.check(tool, args)
    before run(). We give each MCP tool a default_policy and a per-tool
    permission key, so nothing MCP defaults to allow.
  - sandbox confinement: run() confines every string argument through the same
    over-inclusive heuristic bash uses, BEFORE the call is forwarded to the
    server. An out-of-root or credential-named path is refused here, so the
    server never sees it. This is the guarantee: an MCP server can do nothing a
    local tool could not.

## Namespacing

Tool names are `mcp__{label}__{tool}`. Two reasons, both concrete:

  - the Registry rejects duplicate names, and a filesystem server's `read_file`
    would collide with the native `read_file`.
  - it makes the permission key explicit. A policy of
    {'mcp__fs__write_file': 'deny'} is unambiguous about which write it disables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..tools.base import Tool, ToolError


def _root() -> Path:
    return Path(os.environ.get("SA_ROOT", os.getcwd())).resolve()


def _string_values(obj: Any):
    """Yield every string reachable in the argument object.

    Over-inclusive on purpose, matching bash's confiner: a search term that
    happens to contain '/' gets resolved against the root and passes anyway
    (harmless), while a path buried in a nested field still gets checked.
    """
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _string_values(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _string_values(v)


class _MCPArgs:
    """The lightweight validated-args object Registry/permissions expect.

    Registry.execute calls validated.model_dump() to build run()'s kwargs and
    Permissions._describe calls it to show the human what they're approving.
    That is the whole contract, so this is all it needs to implement -- a
    pydantic model would be heavier for no gain when the schema is the server's,
    not ours.
    """

    def __init__(self, data: dict[str, Any]):
        self._data = dict(data)

    def model_dump(self) -> dict[str, Any]:
        return dict(self._data)


def make_mcp_tool(
    client,
    label: str,
    spec: dict[str, Any],
    *,
    path_confine: bool = True,
    default_policy: str = "ask",
) -> type[Tool]:
    """Build one Tool subclass from a discovered MCP tool spec."""
    tool_name = spec["name"]
    namespaced = f"mcp__{label}__{tool_name}"
    input_schema = spec.get("inputSchema") or {"type": "object", "properties": {}}
    base_description = spec.get("description") or f"{tool_name} (via MCP server {label!r})"
    full_description = f"{base_description}\n(external tool from MCP server {label!r})"
    required = list(input_schema.get("required", []))

    # write-shaped tools should never default looser than read-shaped ones;
    # both land on the caller's default_policy, but a name that clearly mutates
    # is forced to ask even if a caller passed allow.
    policy = default_policy
    lowered = tool_name.lower()
    if any(w in lowered for w in ("write", "edit", "move", "delete", "create", "put", "patch")):
        policy = "ask" if default_policy == "allow" else default_policy

    class _MCPTool(Tool):
        name = namespaced
        # description the model reads; keep the server's, note the origin
        description = full_description
        default_policy = policy

        # kept for interface parity; schema()/validate_args() are overridden so
        # this is never actually instantiated by the runtime.
        Args = None  # type: ignore[assignment]

        _mcp_client = client
        _mcp_tool_name = tool_name
        _mcp_input_schema = input_schema
        _mcp_required = required
        _mcp_confine = path_confine

        @classmethod
        def schema(cls) -> dict[str, Any]:
            # the server's real inputSchema, verbatim, under function.parameters
            return {
                "type": "function",
                "function": {
                    "name": cls.name,
                    "description": cls.description.strip(),
                    "parameters": cls._mcp_input_schema,
                },
            }

        @classmethod
        def validate_args(cls, raw: dict[str, Any]) -> _MCPArgs:
            if not isinstance(raw, dict):
                raise ToolError("arguments must be an object")
            missing = [k for k in cls._mcp_required if k not in raw]
            if missing:
                raise ToolError(
                    "invalid arguments -- missing required: " + ", ".join(missing)
                )
            return _MCPArgs(raw)

        def run(self, **kwargs) -> str:
            # CONFINE FIRST, forward second. Every string argument runs through
            # the same path confiner bash uses. An out-of-root path or a
            # credential-named file raises here, before client.call_tool is
            # ever reached, so the MCP server can do nothing read_file could not.
            if type(self)._mcp_confine:
                from ..sandbox import confine_paths
                confine_paths(list(_string_values(kwargs)), _root())
            return type(self)._mcp_client.call_tool(type(self)._mcp_tool_name, kwargs)

    _MCPTool.__name__ = f"MCP_{label}_{tool_name}"
    _MCPTool.__qualname__ = _MCPTool.__name__
    return _MCPTool


def make_mcp_tools(
    client,
    label: str,
    *,
    path_confine: bool = True,
    default_policy: str = "ask",
) -> list[type[Tool]]:
    """Discover a server's tools and map each to a runtime Tool subclass."""
    specs = client.list_tools()
    return [
        make_mcp_tool(
            client, label, spec,
            path_confine=path_confine, default_policy=default_policy,
        )
        for spec in specs
    ]
