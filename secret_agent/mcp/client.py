"""A frameworkless MCP client over the stdio transport.

## Why stdlib and not the official SDK

The Model Context Protocol has an official Python SDK. It is async top to
bottom. This runtime's agent loop is synchronous on purpose (see the project's
DECISIONS), and dragging an event loop in to talk to one local server on a
laptop would buy nothing and cost the thing that makes the rest of the code
readable. So this is JSON-RPC 2.0 over a spawned subprocess's stdin/stdout,
newline-delimited per the MCP stdio transport, in stdlib subprocess + json.

The transport is simpler than it sounds:

  - one JSON-RPC message per line on stdout, no embedded newlines
  - requests carry an integer id; the response echoes that id
  - notifications carry a method and no id and expect no reply

## The handshake

  1. client -> initialize            (protocol version, capabilities, who we are)
  2. server -> result                (its capabilities and info)
  3. client -> notifications/initialized
  4. then tools/list, tools/call, ...

## Reading without deadlocking

A child writing a lot to stderr while nobody drains it fills the pipe buffer
and blocks the child, which then never answers on stdout, which looks like a
hang. So stdout and stderr each get a background reader thread feeding a queue;
the request path just waits on the stdout queue for the id it wants, with a
timeout, and stderr is always being drained in the background and surfaced in
error messages.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2024-11-05"


class MCPError(RuntimeError):
    """The server errored, timed out, or died. Readable on purpose: it becomes
    a tool result the model reads, so a bare code helps no one."""


class MCPStdioClient:
    """One spawned MCP server, spoken to over stdio.

    Use as a context manager, or call connect()/close() by hand:

        with MCPStdioClient(["npx", "-y", "@modelcontextprotocol/server-filesystem", root]) as c:
            for t in c.list_tools():
                ...
            c.call_tool("read_file", {"path": "README.md"})
    """

    def __init__(
        self,
        command: list[str],
        cwd: str | Path | None = None,
        *,
        timeout: float = 30.0,
        client_name: str = "secret-agent",
        client_version: str = "0.1.0",
    ):
        if not command:
            raise ValueError("command must be a non-empty argv list")
        self.command = list(command)
        self.cwd = str(cwd) if cwd is not None else None
        self.timeout = timeout
        self.client_name = client_name
        self.client_version = client_version

        self._proc: subprocess.Popen | None = None
        self._id = 0
        self._inbox: "queue.Queue[dict]" = queue.Queue()
        self._stderr: list[str] = []
        self._readers: list[threading.Thread] = []
        self._server_info: dict[str, Any] = {}

    # --- lifecycle ----------------------------------------------------

    def connect(self) -> "MCPStdioClient":
        if self._proc is not None:
            return self
        try:
            self._proc = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
            )
        except FileNotFoundError as e:
            raise MCPError(
                f"could not start MCP server {self.command[0]!r}: {e}. "
                "Is it installed and on PATH?"
            ) from e

        self._start_reader(self._proc.stdout, self._inbox)
        self._start_stderr_reader(self._proc.stderr)
        self._handshake()
        return self

    def _start_reader(self, stream, inbox: "queue.Queue[dict]") -> None:
        def _pump():
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    inbox.put(json.loads(line))
                except json.JSONDecodeError:
                    # not JSON-RPC; log it so it can be surfaced, don't crash
                    self._stderr.append(f"[non-json stdout] {line}")
            stream.close()

        t = threading.Thread(target=_pump, daemon=True)
        t.start()
        self._readers.append(t)

    def _start_stderr_reader(self, stream) -> None:
        def _pump():
            for line in stream:
                self._stderr.append(line.rstrip("\n"))
            stream.close()

        t = threading.Thread(target=_pump, daemon=True)
        t.start()
        self._readers.append(t)

    def _handshake(self) -> None:
        result = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": self.client_name,
                    "version": self.client_version,
                },
            },
        )
        self._server_info = result if isinstance(result, dict) else {}
        self._notify("notifications/initialized", {})

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        except (OSError, ValueError):
            pass

    def __enter__(self) -> "MCPStdioClient":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    # --- MCP methods --------------------------------------------------

    @property
    def server_info(self) -> dict[str, Any]:
        """serverInfo/capabilities from the initialize result."""
        return self._server_info

    def list_tools(self) -> list[dict[str, Any]]:
        """-> [{'name', 'description', 'inputSchema'}]."""
        result = self._request("tools/list", {})
        tools = result.get("tools", []) if isinstance(result, dict) else []
        return [t for t in tools if isinstance(t, dict) and t.get("name")]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call one tool, return its text content.

        MCP returns a list of content blocks; we join the text ones. A server
        that flags isError is surfaced as an MCPError so the runtime treats it
        the same as any other failing tool result.
        """
        result = self._request(
            "tools/call", {"name": name, "arguments": arguments or {}}
        )
        if not isinstance(result, dict):
            return str(result)

        text = _text_from_content(result.get("content", []))
        if result.get("isError"):
            raise MCPError(text or f"tool {name!r} reported an error")
        return text

    # --- JSON-RPC plumbing -------------------------------------------

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _send(self, message: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise MCPError("MCP client is not connected")
        try:
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise MCPError(f"MCP server closed the connection: {self._why_dead()}") from e

    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict) -> Any:
        req_id = self._next_id()
        self._send(
            {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        )
        return self._await_response(req_id, method)

    def _await_response(self, req_id: int, method: str) -> Any:
        # server-initiated requests/notifications may interleave; skip anything
        # that is not the response to our id.
        deadline_msgs: list[dict] = []
        try:
            while True:
                try:
                    msg = self._inbox.get(timeout=self.timeout)
                except queue.Empty:
                    raise MCPError(
                        f"MCP server did not answer {method!r} within "
                        f"{self.timeout}s. {self._why_dead()}"
                    )
                if msg.get("id") != req_id:
                    # a notification or an unrelated response; ignore it
                    deadline_msgs.append(msg)
                    continue
                if "error" in msg:
                    err = msg["error"]
                    raise MCPError(
                        f"{method} failed: {err.get('message', err)} "
                        f"(code {err.get('code')})"
                    )
                return msg.get("result")
        finally:
            # push back anything we skipped so a later request can see it
            for m in deadline_msgs:
                self._inbox.put(m)

    def _why_dead(self) -> str:
        proc = self._proc
        tail = "\n".join(self._stderr[-10:]).strip()
        code = proc.poll() if proc else None
        bits = []
        if code is not None:
            bits.append(f"process exited with code {code}")
        if tail:
            bits.append(f"stderr:\n{tail}")
        return " ".join(bits) if bits else "no stderr captured"


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(json.dumps(block))
        else:
            parts.append(str(block))
    return "\n".join(parts)
