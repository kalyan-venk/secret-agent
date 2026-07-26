"""The permission layer.

Three outcomes per tool:

    ALLOW  -> run it
    ASK    -> put it to the user; a refusal comes back as a tool result
    DENY   -> refuse without running, and without asking

The shape that matters: a refusal is a *tool result*, not an exception. The
model gets told "the user declined" and can adapt -- suggest an alternative,
explain why it wanted to, ask a question. Blowing up the run because someone
said no to one file write turns a normal interaction into a crash.

Defaults follow reversibility, not danger-in-the-abstract:

    reads      ALLOW   -- undoable by definition, and asking about every read
                          trains the user to hit `y` without looking, which is
                          strictly worse than not asking
    writes     ASK     -- irreversible, but bounded to one resolved path
    bash       ASK     -- irreversible AND unbounded; see the note in
                          tools/shell.py for why it's a different category

Nothing is DENY by default. DENY exists for a caller that wants to hard-off a
tool without unregistering it (so the model still sees it in the schema and
gets a coherent refusal rather than "no such tool").
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

ALLOW = "allow"
ASK = "ask"
DENY = "deny"


@dataclass
class Decision:
    allowed: bool
    reason: str = ""


def _tty_confirm(prompt: str) -> bool:
    if not sys.stdin.isatty():
        # Non-interactive and nobody set auto_approve. Refusing is the only
        # safe read of that: a scripted run that silently gets write access
        # because there was no terminal to ask is how you get a bad morning.
        print(f"\n[permissions] {prompt}\n[permissions] no tty -- refusing")
        return False
    try:
        ans = input(f"\n{prompt} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")


class Permissions:
    def __init__(
        self,
        policies: dict[str, str] | None = None,
        default: str = ASK,
        auto_approve: bool = False,
        confirm: Callable[[str], bool] | None = None,
    ):
        self.policies = dict(policies or {})
        self.default = default
        self.auto_approve = auto_approve
        self.confirm = confirm or _tty_confirm
        # so a run can be audited afterwards
        self.log: list[tuple[str, str, bool]] = []

    def policy_for(self, tool) -> str:
        return self.policies.get(tool.name, getattr(tool, "default_policy", self.default))

    def check(self, tool, args) -> Decision:
        policy = self.policy_for(tool)

        if policy == ALLOW:
            self._note(tool.name, ALLOW, True)
            return Decision(True)

        if policy == DENY:
            self._note(tool.name, DENY, False)
            return Decision(
                False,
                f"the {tool.name} tool is disabled in this session. "
                "Do not try it again; use a different approach.",
            )

        # ASK
        if self.auto_approve:
            self._note(tool.name, "auto", True)
            return Decision(True)

        ok = self.confirm(_describe(tool, args))
        self._note(tool.name, ASK, ok)
        if ok:
            return Decision(True)
        return Decision(
            False,
            f"the user declined the {tool.name} call. "
            "Do not retry it -- explain what you wanted to do and why, or try "
            "something that doesn't need it.",
        )

    def _note(self, name, policy, allowed):
        self.log.append((name, policy, allowed))

    def summary(self) -> str:
        if not self.log:
            return "no permission checks"
        denied = [n for n, _, ok in self.log if not ok]
        s = f"{len(self.log)} checks, {len(denied)} denied"
        if denied:
            s += " (" + ", ".join(sorted(set(denied))) + ")"
        return s


def _describe(tool, args) -> str:
    """What the human is actually approving.

    Show the resolved values, not the tool name. "Allow bash?" is not a
    question anyone can answer correctly; "Allow bash: rm -rf build/" is.
    """
    try:
        d = args.model_dump()
    except AttributeError:
        d = dict(args)
    bits = []
    for k, v in d.items():
        v = str(v)
        if len(v) > 120:
            v = v[:117] + "..."
        bits.append(f"{k}={v}")
    return f"Allow {tool.name}({', '.join(bits)})?"


# Sensible starting point. Passed to Registry(permissions=...).
def default_permissions(auto_approve: bool = False, **kw) -> Permissions:
    return Permissions(
        policies={
            "read_file": ALLOW,
            "list_dir": ALLOW,
            "grep": ALLOW,
            "search_docs": ALLOW,
            "write_file": ASK,
            "bash": ASK,
        },
        default=ASK,
        auto_approve=auto_approve,
        **kw,
    )
