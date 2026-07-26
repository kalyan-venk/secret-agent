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


# Read-only-ish things that are genuinely useful to an agent.
#
# NO INTERPRETERS. python/python3/pytest were here and were a total sandbox
# escape -- see the module docstring. node, ruby, perl, sh, bash, awk, sed -e
# and env belong in the same category and are not coming back. If you want to
# run the test suite, run it yourself; the agent does not need to.
#
# `find` is here under protest. `-exec` is an interpreter in disguise, so it's
# rejected explicitly in BANNED_FLAGS below. That is a blocklist inside an
# allowlist, which is exactly the smell the docstring warns about -- it stays
# only because listing directories is genuinely useful and `list_dir`/`grep`
# cover most of it. Consider dropping `find` entirely.
ALLOWED = {
    "ls", "cat", "head", "tail", "wc", "file", "stat",
    "find", "grep", "rg", "sort", "uniq", "cut", "tr",
    "git", "echo", "pwd", "date", "which",
}

# Flags that turn an allowlisted program back into an arbitrary-execution
# primitive. Not exhaustive and cannot be -- that is the point made in the
# docstring, and the reason this tool asks before running.
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
# note in run(). Redundant with shell=False; kept for the error message.
METACHARS = frozenset({";", "&&", "||", "|", "`", ">", ">>", "<", "&"})

# Substrings that mean the model is writing shell even mid-token: `ls;`,
# `echo `whoami``, `echo $(cat .env)`. Only checked on UNQUOTED tokens.
_EMBEDDED_SHELL = ("`", "$(", ";", "|", ">", "<", "&")

TIMEOUT_S = 30
MAX_OUTPUT = 8000


def _looks_like_path(arg: str) -> bool:
    """Is this argument plausibly naming a file?

    Deliberately over-inclusive. A false positive means a search pattern
    containing '/' gets resolved against the root and passes anyway (harmless).
    A false negative means an unchecked path, which is the bug this exists to
    fix -- so when in doubt, check it.
    """
    if not arg or arg.startswith("-"):
        return False
    return (
        arg.startswith(("/", "~"))
        or "/" in arg
        or ".." in arg
        or (_root() / arg).exists()
    )


def _check_paths(args: list[str], root: Path) -> None:
    """Confine every path-shaped argument, or raise.

    Reuses safe_resolve so bash gets exactly the same containment rule as
    read_file -- traversal, absolute escapes, symlinks out, percent-encoding.
    One implementation, so the two can't drift.
    """
    # imported here rather than at module scope: sandbox imports from
    # tools.base, and a top-level import the other way is a cycle
    from ..sandbox import PathEscape, looks_secret, safe_resolve

    for arg in args:
        if not _looks_like_path(arg):
            continue
        try:
            resolved = safe_resolve(arg, root)
        except PathEscape as e:
            raise ToolError(
                f"{arg!r} is outside the project root, so this command is refused. "
                f"({e})"
            ) from e
        if looks_secret(resolved):
            raise ToolError(
                f"refusing to run this: {arg!r} looks like a credential file. "
                "Same block that applies to read_file."
            )


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

        try:
            argv = shlex.split(command)
        except ValueError as e:
            raise ToolError(f"couldn't parse command: {e}")

        if not argv:
            raise ToolError("empty command")

        # Metachar check runs on TOKENS, after shlex.split, not on the raw
        # string. The raw-string version rejected `grep "a || b" file`, where
        # `||` is a literal search term inside a quoted argument and cannot
        # possibly be an operator.
        #
        # The rule: a token containing whitespace came from quotes, so it is
        # unambiguously DATA and is skipped. A bare token containing an
        # operator is the model writing shell and expecting a shell.
        #
        # To be clear about what this check is for: with shell=False none of
        # these do anything anyway, so this buys no security. It buys a usable
        # error message. `ls ; rm -rf /` otherwise becomes execve looking for a
        # program named ";" and the model learns nothing from the result.
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

        # The check that was missing entirely. Every path-shaped argument is
        # confined to the project root, the same way read_file's is. Without
        # this, `cat /etc/passwd` worked and the claim that permissions and
        # confinement are independent layers was simply false for this tool.
        _check_paths(argv[1:], _root())

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
