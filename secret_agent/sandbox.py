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
#
# ## What this does not do, stated plainly
#
# Name matching only. `prod-token.txt`, `config/api_keys.json` and
# `deploy_secrets.txt` all sail straight through, because nothing here looks
# at content. An external review raised this on 2026-07-25 and explicitly
# declined to score it as a finding, on the grounds that this is a
# belt-and-braces layer and not the confinement boundary. That reading is
# right and it is worth writing down anyway, because the failure mode is a
# reader assuming the opposite: **this list reduces the chance of an accident,
# it does not make credential exposure impossible.** Anything genuinely
# sensitive belongs outside the project root, where confinement applies.
#
# Content-sniffing (entropy, `KEY=`-shaped lines) was considered and rejected:
# it false-positives on test fixtures and example configs, and a guard that
# blocks reading `.env.example` trains the user to switch it off.
#
# Before the review this was worse than described -- `bash` never consulted
# it at all, so `cat .env` returned what `read_file(".env")` refused. See
# MISTAKES.md #14.
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


# ## The one path confiner every tool shares
#
# This is lifted out of tools/shell.py so that bash and any other tool that
# takes model-supplied path arguments (the MCP adapter, for one) run the
# SAME containment rule. MISTAKES.md #13 was a control that existed and was
# tested hard, sitting next to a second tool that never called it -- so the
# lesson is structural: one implementation, imported everywhere, so the two
# cannot drift. If a new tool takes paths, it calls confine_paths and inherits
# traversal, absolute-escape, symlink-out and credential-name refusals for free.


def looks_like_path(arg: str, root: Path) -> bool:
    """Is this argument plausibly naming a file?

    Deliberately over-inclusive. A false positive means a search pattern
    containing '/' gets resolved against the root and passes anyway (harmless).
    A false negative means an unchecked path, which is the bug this exists to
    fix -- so when in doubt, check it.

    A leading '-' still means "not a bare path", but the caller must first pull
    any path hidden inside a flag back out with embedded_paths(); a path smuggled
    as --output=/etc/x reaches here only as its value, never as the flag token.
    """
    if not arg or arg.startswith("-"):
        return False
    return (
        arg.startswith(("/", "~"))
        or "/" in arg
        or ".." in arg
        or (Path(root) / arg).exists()
    )


def embedded_paths(arg: str):
    """Yield path-shaped values hidden inside a flag argument.

    looks_like_path opts out every token starting with '-', which is right for a
    bare flag but wrong when the flag carries a path: `sort --output=/etc/x`,
    `grep --file=/etc/passwd`, or the short joined `sort -o/etc/x` all name a
    file the flag never exposes to the plain check. That is a real escape (a
    write or a read outside the root through an allowlisted program), so pull the
    value out and let it go through the same confinement as a positional path.

    Not a flag, or a flag with no value, yields nothing. The value is still run
    through looks_like_path by the caller, so a non-path value (--color=auto)
    resolves to nothing and is skipped.
    """
    if not arg.startswith("-"):
        return
    if "=" in arg:                       # --output=/x, -f=/x
        value = arg.split("=", 1)[1]
        if value:
            yield value
        return
    if not arg.startswith("--") and len(arg) > 2:   # short joined: -o/etc/x
        yield arg[2:]


def confine_paths(args, root: Path) -> None:
    """Confine every path-shaped argument, or raise.

    Reuses safe_resolve so every caller gets exactly the same containment rule
    as read_file -- traversal, absolute escapes, symlinks out, percent-encoding.
    One implementation, so the callers can't drift.

    Each argument contributes the token itself plus any path embedded in a flag
    (embedded_paths), so `--output=/etc/x` is confined by its value even though
    the token starts with '-'.
    """
    for arg in args:
        if not isinstance(arg, str):
            continue
        for candidate in (arg, *embedded_paths(arg)):
            if not looks_like_path(candidate, root):
                continue
            try:
                resolved = safe_resolve(candidate, root)
            except PathEscape as e:
                raise ToolError(
                    f"{arg!r} is outside the project root, so this command is "
                    f"refused. ({e})"
                ) from e
            if looks_secret(resolved):
                raise ToolError(
                    f"refusing to run this: {arg!r} looks like a credential file. "
                    "Same block that applies to read_file."
                )
