"""bash, on a short leash.

## Read this before trusting anything below

An external review on 2026-07-25 broke this tool completely, and the fixes are
in. What it found, and what it means, because the failure is more instructive
than the design:

    python3 -c 'print(open("/etc/hosts").read())'      -> read outside root
    python3 -c 'open("/tmp/pwned","w").write("OWNED")' -> wrote outside root
    cat /etc/passwd                                    -> read outside root
    cat .env                                           -> read a credential file

`python`, `python3` and `pytest` were on the allowlist. The docstring three
lines above that list said, in these words, "`python -c` is a shell of its
own" -- and then allowlisted it anyway. Every structural defense below was
intact and irrelevant, because the second-stage interpreter was Python, not
/bin/sh. `shell=False` protects you from `/bin/sh`. It does not protect you
from handing the model a different shell.

And the path arguments were never checked at all. `safe_resolve` guarded
read_file and write_file and this tool simply did not call it, so the README's
claim that confinement and permissions are independent layers was false for
bash: approving one bash call escaped the root.

The lesson worth keeping: **an allowlist is only as strong as the least
constrained program on it, and "is this program dangerous" is a much harder
question than it looks.** `find` has `-exec`. `git` has `--exec-path`. Any
interpreter is a total bypass. Enumerating safe binaries is not obviously
easier than enumerating dangerous ones, which was the whole argument for
preferring an allowlist.

## What actually defends this tool now

  1. **shell=False, always.** argv goes to execve. `;`, `&&`, `|`, backticks
     and $() have no meaning there -- no shell is present to interpret them.
     Real, but narrower than it sounds: see above.
  2. **No interpreters on the allowlist.** python/python3/pytest removed. There
     is no argument-level restriction that makes an interpreter safe, so the
     only correct move is not to offer one.
  3. **Every path-shaped argument goes through safe_resolve.** This is the
     defense that was missing. An absolute path outside the root, or anything
     resolving out via `..` or a symlink, is refused before the process spawns.
  4. **looks_secret on resolved arguments**, so `cat .env` is refused the same
     way `read_file(".env")` is.
  5. git restricted to read-only subcommands; cwd pinned; 30s timeout; output
     truncated; environment scrubbed.

## What this still is not

**This is defense in depth, not isolation.** It raises the cost of an escape;
it does not make one impossible. A determined bypass through an allowlisted
binary's own features is still plausible -- these are general-purpose tools
with flags nobody has fully enumerated, and the review demonstrated exactly how
that goes.

Real isolation needs the OS: a seatbelt profile, landlock, or a container.
That is out of scope for a laptop runtime with a human watching, which is why
this tool asks for confirmation by default and must keep doing so.

Env scrubbing is likewise oversold if you read it as credential isolation. HOME
is still forwarded, and `~/.aws/credentials` is a file, not an env var.

## Where the execution actually happens

The validate -> confine -> spawn code used to sit in this file and run in the
agent process. It now lives in `executor/core.py::run_command`, and this tool is
a thin front for it. When SA_EXECUTOR_URL is set, `run()` sends the command to a
separate executor node over HTTP (executor/client.py) instead of spawning a
subprocess here, so untrusted commands do not run in the process that holds the
model keys and the conversation. With SA_EXECUTOR_URL unset it calls the same
`run_command` in-process, so the local path and the remote path share one body
of code and cannot drift in what they allow. See executor/core.py.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from ..config import Config
from ..executor.core import ALLOWED
from .base import Tool, ToolError

_CFG = Config.from_env()


def _root() -> Path:
    return Path(os.environ.get("SA_ROOT", _CFG.root)).resolve()


class Bash(Tool):
    name = "bash"
    description = """Run a single command. No shell operators (; && | > ` $()) --
the command is executed directly, not through a shell, so they will not work.
Only these programs are permitted: """ + ", ".join(sorted(ALLOWED))

    default_policy = "ask"

    class Args(BaseModel):
        command: str = Field(description="one command with its arguments, e.g. 'git status'")

    def run(self, command: str) -> str:
        url = os.environ.get("SA_EXECUTOR_URL")
        if url:
            # Route to the executor node. No local fallback on purpose: if the
            # executor is down bash fails loudly rather than quietly running
            # the command in this process, which is the thing the split exists
            # to stop. See executor/client.py.
            from ..executor.client import ExecutorClient
            return ExecutorClient(url, key=os.environ.get("SA_EXECUTOR_KEY")).execute(command)

        # In-process: same code the executor runs, just here.
        from ..executor.core import run_command
        return run_command(command, _root())


SHELL_TOOLS = [Bash]
