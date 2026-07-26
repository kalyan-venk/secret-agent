"""Filesystem tools.

Every one of these resolves its path through sandbox.safe_resolve before
touching anything. There is no code path in this file that calls open() on a
model-supplied string.

Tools read the root from Config.from_env() at call time rather than taking it
as a constructor argument. That's a compromise I'm not thrilled with -- it
means the root is ambient state -- but Tool subclasses are instantiated by the
registry with no arguments, and threading a config through would mean either
a factory-per-tool or partial application. Noted in DECISIONS.md as the thing
to revisit if this ever grows a second consumer.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import BaseModel, Field

from ..config import Config
from ..sandbox import PathEscape, looks_secret, relative, safe_resolve
from .base import Tool, ToolError

# Read once. Config.from_env() hits os.environ every call and these run in a
# loop; also means a test that sets SA_ROOT before importing gets it.
_CFG = Config.from_env()


def _root() -> Path:
    return Path(os.environ.get("SA_ROOT", _CFG.root)).resolve()


# Anything bigger than this is almost certainly not something the model should
# be reading in full, and it will blow the context window if it does.
MAX_READ_BYTES = 256 * 1024


class ReadFile(Tool):
    name = "read_file"
    description = """Read a UTF-8 text file inside the project and return its contents.
Line numbers are prepended so you can refer to specific lines."""
    default_policy = "allow"

    class Args(BaseModel):
        path: str = Field(description="path relative to the project root, e.g. 'src/main.py'")
        start_line: int | None = Field(
            default=None, description="1-indexed first line to return"
        )
        end_line: int | None = Field(default=None, description="1-indexed last line, inclusive")

    def run(self, path: str, start_line=None, end_line=None) -> str:
        p = safe_resolve(path, _root(), must_exist=True)

        if p.is_dir():
            raise ToolError(f"{path} is a directory -- use list_dir")
        if looks_secret(p):
            raise ToolError(
                f"refusing to read {path}: looks like a credential file. "
                "This is a hard block, not a permission prompt."
            )

        size = p.stat().st_size
        if size > MAX_READ_BYTES:
            raise ToolError(
                f"{path} is {size // 1024}KB, over the {MAX_READ_BYTES // 1024}KB limit. "
                "Use grep to find what you need, or read_file with start_line/end_line."
            )

        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"{path} isn't UTF-8 text -- looks binary")

        lines = text.splitlines()
        lo = (start_line - 1) if start_line else 0
        hi = end_line if end_line else len(lines)
        lo = max(0, lo)
        hi = min(len(lines), hi)
        if lo >= hi:
            raise ToolError(
                f"line range {start_line}-{end_line} is empty; the file has {len(lines)} lines"
            )

        width = len(str(hi))
        out = "\n".join(f"{i + 1:>{width}}  {lines[i]}" for i in range(lo, hi))
        if not text.strip():
            return f"({path} is empty)"
        return out


class WriteFile(Tool):
    name = "write_file"
    description = """Write UTF-8 text to a file inside the project, creating parent
directories if needed. Overwrites existing files. Requires user approval."""
    default_policy = "ask"

    class Args(BaseModel):
        path: str = Field(description="path relative to the project root")
        content: str = Field(description="the full new contents of the file")

    def run(self, path: str, content: str) -> str:
        p = safe_resolve(path, _root())
        if p.is_dir():
            raise ToolError(f"{path} is a directory")
        if looks_secret(p):
            raise ToolError(f"refusing to write {path}: looks like a credential file")

        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        verb = "overwrote" if existed else "created"
        return f"{verb} {relative(p, _root())} ({len(content)} chars, {content.count(chr(10)) + 1} lines)"


# Directories that are never interesting and are always enormous.
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
             ".pytest_cache", ".ruff_cache", "dist", "build", ".chroma"}


class ListDir(Tool):
    name = "list_dir"
    description = """List files and directories at a path inside the project.
Directories are marked with a trailing slash."""
    default_policy = "allow"

    class Args(BaseModel):
        path: str = Field(default=".", description="directory relative to the project root")

    def run(self, path: str = ".") -> str:
        p = safe_resolve(path, _root(), must_exist=True)
        if not p.is_dir():
            raise ToolError(f"{path} is a file, not a directory -- use read_file")

        entries = []
        for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            if child.name in SKIP_DIRS:
                continue
            if child.is_dir():
                entries.append(child.name + "/")
            else:
                try:
                    kb = child.stat().st_size / 1024
                    entries.append(f"{child.name}  ({kb:.1f}KB)")
                except OSError:
                    entries.append(child.name)

        if not entries:
            return f"{relative(p, _root())} is empty"
        return f"{relative(p, _root())}:\n" + "\n".join("  " + e for e in entries)


# A regex with a quantifier applied to a group that already contains one --
# (a+)+, (a*)*, (\d+)* -- can backtrack exponentially. `(a+)+$` against 60
# 'a's followed by anything that fails the match does not finish this decade.
_NESTED_QUANTIFIER = re.compile(r"\([^()]*[+*}][^()]*\)\s*[+*{]")

# Longest line handed to the regex engine. Backtracking blows up in the length
# of the INPUT, so bounding the input bounds the damage even for a pattern the
# check above misses.
MAX_GREP_LINE = 1000


def _reject_catastrophic(pattern: str) -> None:
    """Refuse patterns that can hang the process.

    Found by an external review 2026-07-25: `grep` is `default_policy="allow"`,
    so it runs with no user prompt, and Python's `re` has no timeout and cannot
    be interrupted from another thread. `Grep().run(pattern="(a+)+$", ...)`
    against a 60-character line ran for over ten minutes before being killed.
    Since the pattern is model-supplied -- and could arrive via prompt
    injection in a file the agent just read -- that is a denial of service on
    the agent with no approval step in front of it.

    **This is mitigation, not elimination.** A heuristic on the pattern text
    cannot catch every pathological regex. The real fix is a timeout, and
    Python's `re` module cannot provide one; the honest options are the
    third-party `regex` module (which has `timeout=`) or running the match in
    a subprocess. Neither is worth a dependency here, so: reject the obvious
    shapes, bound the input length, and write down that the guarantee is
    partial.
    """
    if _NESTED_QUANTIFIER.search(pattern):
        raise ToolError(
            f"pattern {pattern!r} nests a quantifier inside a quantified group, "
            "which can backtrack exponentially and hang the search. "
            "Rewrite it without the nesting, e.g. 'a+' instead of '(a+)+'."
        )


class Grep(Tool):
    name = "grep"
    description = """Search file contents for a regular expression, recursively.
Returns matching lines with file paths and line numbers."""
    default_policy = "allow"

    class Args(BaseModel):
        pattern: str = Field(description="a Python regular expression")
        path: str = Field(default=".", description="directory or file to search")
        glob: str = Field(default="*", description="filename filter, e.g. '*.py'")
        max_results: int = Field(default=100, description="cap on returned lines")

    def run(self, pattern: str, path: str = ".", glob: str = "*", max_results: int = 100) -> str:
        root = _root()
        p = safe_resolve(path, root, must_exist=True)

        _reject_catastrophic(pattern)

        try:
            rx = re.compile(pattern)
        except re.error as e:
            # Feed the regex error back rather than crashing. The model wrote
            # the pattern; it can fix the pattern.
            raise ToolError(f"bad regex {pattern!r}: {e}")

        files = [p] if p.is_file() else sorted(p.rglob(glob))
        hits: list[str] = []
        scanned = 0

        for f in files:
            if not f.is_file():
                continue
            if set(f.parts) & SKIP_DIRS:
                continue
            if looks_secret(f):
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable, skip quietly
            scanned += 1
            for n, line in enumerate(text.splitlines(), 1):
                # bound the input, since backtracking blows up in its length.
                # A minified JS file on one 400KB line would be pathological
                # even for a well-behaved pattern.
                if len(line) > MAX_GREP_LINE:
                    line = line[:MAX_GREP_LINE]
                if rx.search(line):
                    trimmed = line.strip()
                    if len(trimmed) > 200:
                        trimmed = trimmed[:200] + "..."
                    hits.append(f"{relative(f, root)}:{n}: {trimmed}")
                    if len(hits) >= max_results:
                        break
            if len(hits) >= max_results:
                hits.append(f"... stopped at {max_results} results")
                break

        if not hits:
            return f"no matches for {pattern!r} in {scanned} files"
        return "\n".join(hits)


FS_TOOLS = [ReadFile, WriteFile, ListDir, Grep]
