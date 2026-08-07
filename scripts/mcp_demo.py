"""Demonstrate an external MCP server driven through the runtime's guardrails.

    SA_ROOT=$PWD .venv/bin/python scripts/mcp_demo.py

Connects to the reference filesystem MCP server rooted at this repo, discovers
its tools and maps them into the runtime's dispatch + permission + sandbox
layers, then runs three probes:

  1. read an in-root file            -> allowed, forwarded to the server
  2. read /etc/passwd                -> refused (outside root), server never called
  3. read .env                       -> refused (credential), same as read_file

It prints the two Capture facts: the number of MCP-server tools integrated, and
that a sandbox-blocked action stays blocked when it arrives via MCP.

Requires node/npx. Everything else is stdlib + this package.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secret_agent.mcp.adapter import make_mcp_tools
from secret_agent.mcp.client import MCPStdioClient
from secret_agent.tools.base import ToolError


def _pick(tools, *candidates):
    for c in candidates:
        if c in tools:
            return tools[c]
    raise SystemExit(f"server exposed none of {candidates}; has {sorted(tools)}")


def main() -> int:
    if shutil.which("npx") is None:
        print("npx not found; install node to run this demo", file=sys.stderr)
        return 1

    root = Path(os.environ.get("SA_ROOT", os.getcwd())).resolve()
    os.environ["SA_ROOT"] = str(root)

    # an in-root file the server can actually read
    corpus = root / "corpus"
    in_root_rel = None
    if corpus.is_dir():
        docs = sorted(p for p in corpus.glob("*.txt") if p.name != "overview-index.txt")
        if docs:
            in_root_rel = str(docs[0].relative_to(root))
    if in_root_rel is None:
        in_root_rel = "README.md"

    client = MCPStdioClient(
        ["npx", "-y", "@modelcontextprotocol/server-filesystem", str(root)],
        cwd=str(root),
        timeout=90,
    ).connect()

    # a throwaway credential file so probe 3 exercises the real refusal path;
    # removed in finally so it never lingers in the repo.
    env_path = root / ".env"
    created_env = False

    passed = 0
    try:
        tools = {t.name: t for t in make_mcp_tools(client, "fs")}
        n = len(tools)
        print(f"MCP server: {client.server_info.get('serverInfo', {})}")
        print(f"MCP tools integrated through the runtime: {n}")
        print("names: " + ", ".join(sorted(tools)))
        print()

        reader = _pick(tools, "mcp__fs__read_text_file", "mcp__fs__read_file")

        # probe 1: in-root read, allowed and forwarded
        try:
            out = reader().run(path=in_root_rel)
            ok = bool(out) and "outside the project root" not in out
            print(f"[{'PASS' if ok else 'FAIL'}] probe 1  in-root read of {in_root_rel!r} "
                  f"-> {len(out)} chars returned")
            passed += ok
        except ToolError as e:
            print(f"[FAIL] probe 1  in-root read raised: {e}")

        # probe 2: /etc/passwd, refused before the server is called
        try:
            reader().run(path="/etc/passwd")
            print("[FAIL] probe 2  /etc/passwd was NOT refused")
        except ToolError as e:
            ok = "outside the project root" in str(e)
            print(f"[{'PASS' if ok else 'FAIL'}] probe 2  /etc/passwd refused "
                  f"before forwarding ({_first_line(e)})")
            passed += ok

        # probe 3: .env, refused as a credential exactly like read_file
        if not env_path.exists():
            env_path.write_text("SECRET_KEY=demo-not-a-real-secret\n")
            created_env = True
        try:
            reader().run(path=".env")
            print("[FAIL] probe 3  .env was NOT refused")
        except ToolError as e:
            ok = "credential" in str(e)
            print(f"[{'PASS' if ok else 'FAIL'}] probe 3  .env refused as credential "
                  f"({_first_line(e)})")
            passed += ok

        print()
        print(f"probes passed: {passed}/3")
        print(f"MCP tools integrated: {n}")
        print("a sandbox-blocked action stays blocked when it arrives via MCP: "
              + ("YES" if passed == 3 else "NO"))
        return 0 if passed == 3 else 1
    finally:
        if created_env:
            try:
                env_path.unlink()
            except OSError:
                print(f"warning: could not remove {env_path}", file=sys.stderr)
        client.close()


def _first_line(e) -> str:
    return str(e).splitlines()[0]


if __name__ == "__main__":
    raise SystemExit(main())
