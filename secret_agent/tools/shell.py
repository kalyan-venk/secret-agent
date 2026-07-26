"""bash, on a short leash.

Why this tool is a different category from write_file even though both mutate:

  write_file's blast radius is one path, and that path goes through
  safe_resolve, so containment is *checkable*. I can look at the resolved path
  and know what will be touched.

  bash's blast radius is whatever the process decides to do. `curl` exfiltrates,
  `git push` publishes, `rm` deletes outside the root, `python -c` is a shell of
  its own. There is no argument I can inspect that bounds the effect, because
  the effect isn't in the arguments -- it's in the program.

So the defenses here are structural rather than analytical:

  1. shell=False, always. The command is shlex.split into an argv list and
     handed to execve. `;`, `&&`, `|`, backticks and $() have no meaning to
     execve -- there is no shell present to interpret them. This is the real
     defense and it holds even if every check below is wrong.
  2. An allowlist of executables. Not a blocklist: a blocklist is a claim that
     you enumerated every dangerous binary on an unknown machine, which is not
     a claim anyone can make. An allowlist is a claim that you enumerated the
     ones you need, which is easy and stays true.
  3. Shell metacharacters rejected explicitly anyway. Redundant with (1), but
     it turns a silently-weird result ("no such file: foo;") into an error
     message the model can learn from, and it makes the intent visible to
     anyone reading the code.
  4. cwd pinned to the project root, timeout, output truncated.

Even with all that: this tool asks for confirmation by default and should stay
that way.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from ..config import Config
from .base import Tool, ToolError

_CFG = Config.from_env()


def _root() -> Path:
    return Path(os.environ.get("SA_ROOT", _CFG.root)).resolve()


# Read-only-ish things that are genuinely useful to an agent. `git` is in
# here and it's the one I went back and forth on -- `git push` is a publish
# and `git reset --hard` destroys work. Subcommand filtering below.
ALLOWED = {
    "ls", "cat", "head", "tail", "wc", "file", "stat",
    "find", "grep", "rg", "sort", "uniq", "cut", "tr",
    "python", "python3", "pytest", "git", "echo", "pwd", "date", "which",
}

# git subcommands that are safe to run unattended
GIT_READONLY = {"status", "log", "diff", "show", "branch", "ls-files",
                "rev-parse", "describe", "blame", "remote"}

# Redundant with shell=False. Kept for the error message. See docstring (3).
METACHARS = (";", "&&", "||", "|", "`", "$(", ">", ">>", "<", "\n", "&")

TIMEOUT_S = 30
MAX_OUTPUT = 8000


class Bash(Tool):
    name = "bash"
    description = """Run a single command. No shell operators (; && | > ` $()) --
the command is executed directly, not through a shell, so they will not work.
Only these programs are permitted: """ + ", ".join(sorted(ALLOWED))

    default_policy = "ask"

    class Args(BaseModel):
        command: str = Field(description="one command with its arguments, e.g. 'git status'")

    def run(self, command: str) -> str:
        command = command.strip()
        if not command:
            raise ToolError("empty command")

        for m in METACHARS:
            if m in command:
                raise ToolError(
                    f"{m!r} is not allowed -- this runs without a shell, so operators "
                    "have no effect. Run one command, or make several separate calls."
                )

        try:
            argv = shlex.split(command)
        except ValueError as e:
            raise ToolError(f"couldn't parse command: {e}")

        if not argv:
            raise ToolError("empty command")

        prog = Path(argv[0]).name  # /usr/bin/rm -> rm, so a path can't sneak past
        if prog not in ALLOWED:
            raise ToolError(
                f"{prog!r} is not on the allowlist. Permitted: "
                f"{', '.join(sorted(ALLOWED))}"
            )

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
                cwd=_root(),
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
                # Don't inherit the parent env wholesale -- it has API keys in
                # it on most machines and a subprocess doesn't need them.
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


SHELL_TOOLS = [Bash]
