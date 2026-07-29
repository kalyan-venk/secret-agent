"""MCP adapter: an external server's tools become ordinary Registry tools, and
every guardrail the native tools go through governs them too.

The sharp claim being proven here is the second half of the Capture metric:
**a sandbox-blocked action stays blocked when it arrives via MCP.** The offline
tests use a stub client and assert that a blocked path raises BEFORE
client.call_tool is ever reached (call_count stays zero), so the server never
sees the request. The .env test mirrors read_file('.env') exactly, which is the
proof that MCP can do nothing a local tool could not: the real filesystem
server WOULD happily read a .env inside its root, and our layer refuses it the
same way it refuses read_file.

The integration test (marker `mcp`, needs node/npx) spawns the reference
filesystem server and shows an in-root read succeed and /etc/passwd refused by
our layer.
"""

import os
import shutil
from pathlib import Path

import pytest

from secret_agent.mcp.adapter import make_mcp_tool, make_mcp_tools
from secret_agent.parsing import ToolCall
from secret_agent.permissions import Permissions, default_permissions
from secret_agent.tools.base import ToolError
from secret_agent.tools.fs import ReadFile
from secret_agent.tools.registry import Registry


# --- a stub MCP client, so the offline suite needs no node -------------


class StubClient:
    def __init__(self, specs):
        self._specs = specs
        self.calls = []  # (name, arguments) for every call_tool that got through

    def list_tools(self):
        return self._specs

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return f"stub-result for {name} {arguments}"


FS_SPECS = [
    {
        "name": "read_file",
        "description": "Read a file's contents",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
]


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "README.md").write_text("# hi\nin-root content\n")
    (root / ".env").write_text("SECRET_KEY=hunter2\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "passwd.txt").write_text("root:x:0:0\n")
    monkeypatch.setenv("SA_ROOT", str(root))
    return root


@pytest.fixture
def stub_tools():
    stub = StubClient(FS_SPECS)
    tools = make_mcp_tools(stub, "fs")
    by_name = {t.name: t for t in tools}
    return stub, tools, by_name


# --- (a) namespacing + schema pass-through -----------------------------


def test_names_are_namespaced_to_avoid_registry_collision(stub_tools):
    _, _, by_name = stub_tools
    assert set(by_name) == {"mcp__fs__read_file", "mcp__fs__write_file"}


def test_schema_passes_the_servers_inputschema_through_verbatim(stub_tools):
    _, _, by_name = stub_tools
    schema = by_name["mcp__fs__write_file"].schema()
    params = schema["function"]["parameters"]
    assert params is FS_SPECS[1]["inputSchema"] or params == FS_SPECS[1]["inputSchema"]
    assert params["required"] == ["path", "content"]


def test_a_namespaced_mcp_tool_is_a_normal_registry_tool(stub_tools):
    stub, tools, _ = stub_tools
    reg = Registry(tools, permissions=default_permissions(auto_approve=True))
    assert "mcp__fs__read_file" in reg.names
    assert any(s["function"]["name"] == "mcp__fs__read_file" for s in reg.schemas())


# --- (b) confinement blocks BEFORE the server is called ----------------


@pytest.mark.parametrize("bad", ["/etc/passwd", "../../etc/passwd", "../outside/passwd.txt"])
def test_out_of_root_path_is_refused_before_forwarding(project, stub_tools, bad):
    stub, _, by_name = stub_tools
    with pytest.raises(ToolError, match="outside the project root"):
        by_name["mcp__fs__read_file"]().run(path=bad)
    # the whole point: the server never saw it
    assert stub.calls == []


# --- (c) credential-name refusal mirrors read_file('.env') exactly -----


def test_dot_env_is_refused_as_credential_and_never_forwarded(project, stub_tools):
    stub, _, by_name = stub_tools
    with pytest.raises(ToolError, match="credential"):
        by_name["mcp__fs__read_file"]().run(path=".env")
    assert stub.calls == []


def test_mcp_env_refusal_matches_native_read_file(project, stub_tools):
    # read_file('.env') is a hard credential block. The MCP read_file must be
    # the same block, or MCP would be able to do something a local tool cannot.
    _, _, by_name = stub_tools
    with pytest.raises(ToolError, match="credential"):
        ReadFile().run(path=".env")
    with pytest.raises(ToolError, match="credential"):
        by_name["mcp__fs__read_file"]().run(path=".env")


# --- (d) permission policy governs MCP tools ---------------------------


def test_deny_policy_refuses_an_mcp_tool_without_running_it(project, stub_tools):
    stub, tools, _ = stub_tools
    perms = Permissions({"mcp__fs__write_file": "deny"})
    reg = Registry(tools, permissions=perms)
    r = reg.execute(
        ToolCall(name="mcp__fs__write_file",
                 arguments={"path": "a.txt", "content": "x"})
    )
    assert not r.ok and "disabled" in r.content
    assert stub.calls == []


def test_in_root_call_routes_through_registry_to_the_server(project, stub_tools):
    # the happy path: validate -> permission -> confine -> forward
    stub, tools, _ = stub_tools
    reg = Registry(tools, permissions=default_permissions(auto_approve=True))
    r = reg.execute(ToolCall(name="mcp__fs__read_file", arguments={"path": "README.md"}))
    assert r.ok
    assert stub.calls == [("read_file", {"path": "README.md"})]


def test_missing_required_arg_is_a_readable_error(project, stub_tools):
    stub, tools, _ = stub_tools
    reg = Registry(tools, permissions=default_permissions(auto_approve=True))
    r = reg.execute(ToolCall(name="mcp__fs__write_file", arguments={"path": "a.txt"}))
    assert not r.ok and "content" in r.content
    assert stub.calls == []


def test_write_shaped_tool_never_defaults_to_allow():
    stub = StubClient(FS_SPECS)
    tools = {t.name: t for t in make_mcp_tools(stub, "fs", default_policy="allow")}
    # even asked for allow, a write-shaped tool is forced to ask
    assert tools["mcp__fs__write_file"].default_policy == "ask"
    assert tools["mcp__fs__read_file"].default_policy == "allow"


# --- integration: the real reference filesystem server -----------------


@pytest.mark.mcp
def test_real_filesystem_server_in_root_ok_out_of_root_refused(tmp_path, monkeypatch):
    if shutil.which("npx") is None:
        pytest.skip("npx not available")

    from secret_agent.mcp.client import MCPStdioClient

    root = tmp_path / "proj"
    root.mkdir()
    (root / "hello.txt").write_text("hi there from in root")
    monkeypatch.setenv("SA_ROOT", str(root))

    client = MCPStdioClient(
        ["npx", "-y", "@modelcontextprotocol/server-filesystem", str(root)],
        cwd=str(root),
        timeout=90,
    ).connect()
    try:
        tools = {t.name: t for t in make_mcp_tools(client, "fs")}
        assert len(tools) >= 10  # the reference server exposes ~14
        assert "mcp__fs__read_text_file" in tools

        # in-root read succeeds, through our confinement and out to the server
        out = tools["mcp__fs__read_text_file"]().run(path="hello.txt")
        assert "hi there from in root" in out

        # out-of-root read is refused by OUR layer, before the server is asked
        with pytest.raises(ToolError, match="outside the project root"):
            tools["mcp__fs__read_text_file"]().run(path="/etc/passwd")
    finally:
        client.close()
