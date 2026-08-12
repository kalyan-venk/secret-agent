"""Prove the bash tool runs commands on a separate executor process.

    SA_ROOT=$PWD .venv/bin/python scripts/executor_demo.py

Starts a real uvicorn executor node on a free localhost port, points the bash
tool at it (SA_EXECUTOR_URL), and runs commands through the ordinary
Bash().run() the agent uses. No LLM involved -- this is about the process split,
not the model.

What it shows:

  1. the executor node reports a different PID than this process, so a command
     that runs "through bash" is genuinely running over there
  2. an allowed command's output comes back over HTTP
  3. `cat /etc/passwd` is refused on the executor side, so the confinement moved
     with the execution and did not stay behind in this process
  4. kill the executor and the same bash call fails "unreachable" -- there is no
     silent local fallback, which is the whole reason for the split

Needs the server extra (fastapi/uvicorn): pip install -e ".[server]".
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secret_agent.tools.base import ToolError
from secret_agent.tools.shell import Bash


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_health(url: str, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError as e:
            last = e
        time.sleep(0.25)
    raise SystemExit(f"executor did not come up at {url}: {last}")


def main() -> int:
    root = Path(os.environ.get("SA_ROOT", os.getcwd())).resolve()
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    key = "demo-executor-key"

    env = dict(os.environ)
    env["SA_EXECUTOR_KEY"] = key
    env["SA_ROOT"] = str(root)
    env["SA_EXECUTOR_LOG"] = "1"  # make the node log which commands it ran

    print(f"orchestrator pid: {os.getpid()}")
    print(f"starting executor node on {url}, root={root}\n")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "secret_agent.executor.service:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    passed = 0
    try:
        health = _wait_health(url)
        print(f"executor /health: {health}")
        same = health["pid"] == os.getpid()
        print(f"[{'FAIL' if same else 'PASS'}] executor pid {health['pid']} "
              f"is a different process than this one ({os.getpid()})")
        passed += not same

        # point the bash tool at the executor node
        os.environ["SA_EXECUTOR_URL"] = url
        os.environ["SA_EXECUTOR_KEY"] = key

        # 2: allowed command, output returns over HTTP
        out = Bash().run(command="echo ran-on-the-executor")
        ok = "ran-on-the-executor" in out
        print(f"[{'PASS' if ok else 'FAIL'}] bash 'echo' ran remotely and "
              f"returned {out!r}")
        passed += ok

        # 3: confinement enforced on the executor side
        try:
            Bash().run(command="cat /etc/passwd")
            print("[FAIL] 'cat /etc/passwd' was NOT refused")
        except ToolError as e:
            ok = "outside the project root" in str(e)
            print(f"[{'PASS' if ok else 'FAIL'}] 'cat /etc/passwd' refused on the "
                  f"executor side ({str(e).splitlines()[0]})")
            passed += ok

        # give the executor a moment to flush its request logs, then show them
        time.sleep(0.3)
    finally:
        proc.terminate()
        try:
            logs, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            logs, _ = proc.communicate()

    print("\n--- executor process log (its own stdout, proving the commands "
          "landed there) ---")
    for line in (logs or "").splitlines():
        if "executor pid=" in line:
            print("  " + line.strip())

    # 4: executor is down now; the same call must fail, not fall back to local
    print("\nexecutor stopped. the same bash call must now fail, not run locally:")
    try:
        Bash().run(command="echo ran-on-the-executor")
        print("[FAIL] bash ran with the executor down -- there is a local fallback")
    except ToolError as e:
        ok = "unreachable" in str(e)
        print(f"[{'PASS' if ok else 'FAIL'}] bash refused with the executor down "
              f"({str(e).splitlines()[0]})")
        passed += ok

    print(f"\nchecks passed: {passed}/4")
    return 0 if passed == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
