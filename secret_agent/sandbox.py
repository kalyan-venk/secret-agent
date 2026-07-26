"""Path confinement.

Every path that reaches a file tool came out of a language model, which means
it is untrusted input in exactly the sense a URL query parameter is untrusted
input. It does not matter that the model is "trying to help" -- it will
happily pass along a path a user asked it to, and prompt injection in a file
it just read is a real way for that to happen without anyone typing it.

The rule: resolve to canonical form, then check containment. Never hand the
raw string to open().

Things this stops, and how:

  ../../../../etc/passwd    resolve() collapses it, lands outside root, caught
  /etc/passwd               absolute, lands outside root, caught
  a_symlink_to_tmp          resolve() FOLLOWS the link, lands outside, caught
  ~/. ssh/id_rsa            expanduser is deliberately NOT called, so this is
                            just a directory literally named "~" -- inert
  %2e%2e%2fetc              see below

The URL-encoded case is worth being precise about, because it's the one it's
easy to bluff. Nothing here URL-decodes anything. `%2e%2e%2f` reaching open()
is a request for a file whose name contains those characters; it is not
traversal and never becomes traversal. It's rejected below anyway as a signal
(a model sending percent-encoded paths is confused or being steered), but the
honest answer to "how do you stop encoded traversal" is "by never decoding",
not "by blocklisting the string".

What this does NOT stop, stated rather than papered over:

  - TOCTOU. Between resolve() and open() a symlink could be swapped. Closing
    that needs openat2/O_NOFOLLOW and a fd-relative walk. For a laptop agent
    with a human watching, out of scope; in a multi-tenant service it would
    not be.
  - A hardlink inside the root pointing at an inode outside it. resolve()
    can't see that -- hardlinks have no path to follow. Real, unfixed, and
    requires an attacker who already has write access to the root.
"""

from __future__ import annotations

import os
from pathlib import Path

from .tools.base import ToolError


class PathEscape(ToolError):
    """Path resolved outside the project root."""


# Rejected on sight. None of these are load-bearing for security (see the
# module docstring on encoding) -- they're a tripwire that says the caller
# is doing something other than naming a file.
SUSPICIOUS = ("%2e", "%2f", "%5c", "\x00")


def safe_resolve(raw: str, root: Path, *, must_exist: bool = False) -> Path:
    """raw (untrusted) -> an absolute Path guaranteed to sit inside root.

    Raises PathEscape otherwise. The error text is deliberately readable by
    the model: it gets fed back as a tool result and a model that understands
    why it was refused will usually correct itself, where a bare "denied"
    makes it try the same thing three more times.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise PathEscape("path must be a non-empty string")

    low = raw.lower()
    for s in SUSPICIOUS:
        if s in low:
            raise PathEscape(
                f"path contains {s!r}, which is not decoded here. "
                "Pass a plain relative path like 'src/main.py'."
            )

    # root itself has to be canonical or the comparison is meaningless. On
    # macOS /tmp is a symlink to /private/tmp, so an unresolved root makes
    # every path under a tmpdir look like an escape. Cost me a while in tests.
    root = Path(root).resolve()

    p = Path(raw)
    # NOT expanduser(). "~" stays a literal directory name; the model does not
    # get to address the home directory.
    if p.is_absolute():
        candidate = p
    else:
        candidate = root / p

    # strict=False so a not-yet-created file (write_file) still resolves.
    # Non-existent components stay lexical, which is fine -- the parent chain
    # is what symlinks live in and that part IS resolved.
    resolved = candidate.resolve(strict=False)

    if not _contained(resolved, root):
        raise PathEscape(
            f"{raw!r} resolves to {resolved}, which is outside the project root "
            f"({root}). Only paths inside the project are allowed."
        )

    if must_exist and not resolved.exists():
        raise ToolError(f"no such file or directory: {raw}")

    return resolved


def _contained(p: Path, root: Path) -> bool:
    # is_relative_to is 3.9+ and does the right thing on the string form.
    # `root` counts as inside itself, which list_dir(".") relies on.
    return p == root or p.is_relative_to(root)


def relative(p: Path, root: Path) -> str:
    """For display. Absolute paths in tool output leak the machine's layout
    into the model's context for no benefit."""
    try:
        return str(p.relative_to(Path(root).resolve())) or "."
    except ValueError:
        return str(p)


# Files that exist in most repos and that a model has no business reading.
# Belt and braces on top of confinement -- these are all INSIDE the root, so
# path checking doesn't touch them.
SECRET_NAMES = {
    ".env", ".env.local", ".env.production",
    "id_rsa", "id_ed25519", ".netrc", ".pgpass",
    "credentials", "secrets.yaml", "secrets.yml",
}
SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx")


def looks_secret(p: Path) -> bool:
    name = p.name.lower()
    if name in SECRET_NAMES:
        return True
    if name.endswith(SECRET_SUFFIXES):
        return True
    # .git/config holds remote URLs which sometimes carry tokens
    return ".git" in p.parts and p.name == "config"
