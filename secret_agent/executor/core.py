"""The command-execution core, lifted out of the orchestrator process.

This is the code that actually spawns a subprocess for the `bash` tool. It used
to live inside `tools/shell.py` and run in the same process as the agent loop.
After the 2026-07-25 review turned `python -c` into a full sandbox escape, the
one thing worth doing on top of the in-process guardrails was to stop running
untrusted commands in the orchestrator process at all. So the whole validate ->
confine -> spawn path is here, as a plain function with no web framework and no
loop attached to it, and `executor/service.py` wraps it in an HTTP endpoint that
can run on a different box.

`run_command(command, root)` is the single implementation. The in-process `Bash`
tool calls it directly when no executor node is configured, and the HTTP service
calls the same function on the other side of the wire. One body of code, so the
local path and the remote path cannot drift apart in what they allow. That is
the same rule the path confinement follows (see sandbox.confine_paths).

None of the guardrails below are isolation. They raise the cost of an escape;
the OS is what actually contains it. Running this as a separate service is the
step toward real isolation: the executor can be a locked-down box (its own root,
its own network rules, a container or a jail) and a compromise of the command
tool is a compromise of that box, not of the process holding the model keys and
the conversation. The guardrails are still here because defense in depth is
cheap and the review showed exactly how a single layer fails.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from ..sandbox import confine_paths
from ..tools.base import ToolError


# Read-only-ish things that are genuinely useful to an agent.
#
# NO INTERPRETERS. python/python3/pytest were here and were a total sandbox
# escape -- see MISTAKES.md #12. node, ruby, perl, sh, bash, awk, sed -e and
# env belong in the same category and are not coming back. If you want to run
# the test suite, run it yourself; the agent does not need to.
#
# `find` is here under protest. `-exec` is an interpreter in disguise, so it's
# rejected explicitly in BANNED_FLAGS below. That is a blocklist inside an
# allowlist, which is exactly the smell the shell.py docstring warns about -- it
# stays only because listing directories is genuinely useful and `list_dir`/
# `grep` cover most of it. Consider dropping `find` entirely.
ALLOWED = {
    "ls", "cat", "head", "tail", "wc", "file", "stat",
    "find", "grep", "rg", "sort", "uniq", "cut", "tr",
    "git", "echo", "pwd", "date", "which",
}

# Flags that turn an allowlisted program back into an arbitrary-execution
# primitive. Not exhaustive and cannot be -- that is the point made in the
# shell.py docstring, and the reason the bash tool asks before running.
BANNED_FLAGS = {
    "-exec", "-execdir", "-ok", "-okdir",       # find
    "-fprint", "-fprintf", "-fls",              # find, writes anywhere
    "--exec-path", "--upload-pack",             # git
    "-c", "--command",                          # anything shell-shaped
    "--use-compress-program",                   # git/tar
}

# git subcommands that are safe to run unattended
GIT_READONLY = {"status", "log", "diff", "show", "branch", "ls-files",
                "rev-parse", "describe", "blame", "remote"}

# Checked per-token after shlex.split, NOT against the raw string -- see the
# note in run_command(). Redundant with shell=False; kept for the error message.
METACHARS = frozenset({";", "&&", "||", "|", "`", ">", ">>", "<", "&"})

# Substrings that mean the model is writing shell even mid-token: `ls;`,
# `echo `whoami``, `echo $(cat .env)`. Only checked on UNQUOTED tokens.
_EMBEDDED_SHELL = ("`", "$(", ";", "|", ">", "<", "&")

TIMEOUT_S = 30
MAX_OUTPUT = 8000


def run_command(command: str, root: Path) -> str:
    """Validate, confine, and run one command inside `root`.

    Raises ToolError for anything refused (bad program, banned flag, path
    escape, credential file, shell syntax). Returns the command's output as a
    string on success, including the `[exit N]` prefix for a non-zero exit,
    which is information the model needs rather than a crash.

    `root` is passed in, never read from ambient state, so the executor service
    can pin it to whatever this node is allowed to touch.
    """
    root = Path(root).resolve()
    command = command.strip()
    if not command:
        raise ToolError("empty command")

    try:
        argv = shlex.split(command)
    except ValueError as e:
        raise ToolError(f"couldn't parse command: {e}")

    if not argv:
        raise ToolError("empty command")

    # Metachar check runs on TOKENS, after shlex.split, not on the raw string.
    # The raw-string version rejected `grep "a || b" file`, where `||` is a
    # literal search term inside a quoted argument and cannot be an operator.
    #
    # The rule: a token containing whitespace came from quotes, so it is
    # unambiguously DATA and is skipped. A bare token containing an operator is
    # the model writing shell and expecting a shell.
    #
    # With shell=False none of these do anything anyway, so the check buys no
    # security at all -- it buys a usable error message.
    for tok in argv:
        if any(c.isspace() for c in tok):
            continue  # quoted -> data
        if tok in METACHARS or any(m in tok for m in _EMBEDDED_SHELL):
            raise ToolError(
                f"{tok!r} looks like shell syntax. This runs without a shell, "
                "so operators and substitutions have no effect. Run one "
                "command, or make several separate calls. To search for the "
                "characters literally, quote the whole pattern."
            )

    prog = Path(argv[0]).name  # /usr/bin/rm -> rm, so a path can't sneak past
    if prog not in ALLOWED:
        raise ToolError(
            f"{prog!r} is not on the allowlist. Permitted: "
            f"{', '.join(sorted(ALLOWED))}. "
            "Interpreters (python, node, sh) are permanently excluded -- "
            "they are arbitrary code execution regardless of arguments."
        )

    for a in argv[1:]:
        flag = a.split("=", 1)[0]
        if flag in BANNED_FLAGS:
            raise ToolError(
                f"{flag!r} is not allowed -- it turns {prog} into a way to "
                "run arbitrary programs."
            )

    # Every path-shaped argument is confined to the root, the same way
    # read_file's is. Without this, `cat /etc/passwd` worked and the claim that
    # permissions and confinement are independent layers was simply false for
    # this tool. See MISTAKES.md #13.
    confine_paths(argv[1:], root)

    if prog == "git":
        sub = next((a for a in argv[1:] if not a.startswith("-")), None)
        if sub not in GIT_READONLY:
            raise ToolError(
                f"git {sub!r} is not permitted. Read-only subcommands only: "
                f"{', '.join(sorted(GIT_READONLY))}"
            )

    try:
        proc = subprocess.run(
            argv,
            shell=False,          # the actual defense. do not change this.
            cwd=root,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            # Don't inherit the parent env wholesale -- it has API keys in it on
            # most machines and a subprocess doesn't need them.
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": os.environ.get("HOME", ""),
                "LANG": os.environ.get("LANG", "C.UTF-8"),
            },
        )
    except FileNotFoundError:
        raise ToolError(f"{prog}: not found on this machine")
    except subprocess.TimeoutExpired:
        raise ToolError(f"{prog} didn't finish in {TIMEOUT_S}s and was killed")

    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    out = out.strip()

    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT] + f"\n... truncated at {MAX_OUTPUT} chars"

    if proc.returncode != 0:
        # Non-zero is information, not a crash. `grep` returns 1 for "no
        # matches" and the model needs to see that rather than an error.
        return f"[exit {proc.returncode}]\n{out}" if out else f"[exit {proc.returncode}, no output]"

    return out or "(no output)"
