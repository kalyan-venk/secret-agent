"""Message history.

The thing to hold onto: **there is no session on the server.** /api/chat is
stateless. Every single turn re-sends the entire history from scratch. The
model doesn't "remember" turn 1 -- turn 1 is physically in the payload of
turn 3.

Two consequences that drive the rest of the repo:
  1. This list IS the memory. If the process dies it's gone, which is why
     save()/load() exist below.
  2. The payload grows every turn, and it grows fastest when tools return
     big blobs. That's Phase 5's whole reason to exist.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    # tool messages carry these; everything else leaves them None
    tool_call_id: str | None = None
    name: str | None = None
    # not sent to the model -- bookkeeping for context.py
    pinned: bool = False
    created_at: float = field(default_factory=time.time)

    def to_wire(self) -> dict[str, Any]:
        """Ollama's shape. Note it does NOT want tool_call_id -- it just wants
        role=tool with the content, and it figures out the association from
        ordering. Anthropic wants tool_use_id. If a second client ever lands,
        this method moves onto the client."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.role == "tool" and self.name:
            d["name"] = self.name
        return d

    def char_len(self) -> int:
        return len(self.content) + len(self.role) + 8  # +8 for the wrapper-ish


def new_call_id() -> str:
    # short ids because they end up in printed transcripts and full uuids are
    # unreadable there
    return "call_" + uuid.uuid4().hex[:8]


class Conversation:
    def __init__(self, system: str | None = None):
        self.messages: list[Message] = []
        if system:
            self.add_system(system)
        # incremented whenever context.py rewrites history, for debugging
        self.compactions = 0

    # --- building ---------------------------------------------------

    def add_system(self, text: str) -> Message:
        m = Message(role="system", content=text, pinned=True)
        # system prompt goes first, always. If one already exists we replace
        # rather than append -- two system messages confuses small models
        # badly (llama3.1 started answering as if the second one was a user).
        existing = next((i for i, x in enumerate(self.messages) if x.role == "system"), None)
        if existing is not None:
            self.messages[existing] = m
        else:
            self.messages.insert(0, m)
        return m

    def add_user(self, text: str) -> Message:
        m = Message(role="user", content=text)
        self.messages.append(m)
        return m

    def add_assistant(self, text: str) -> Message:
        m = Message(role="assistant", content=text)
        self.messages.append(m)
        return m

    def add_tool_result(self, call_id: str, name: str, result: str) -> Message:
        m = Message(role="tool", content=result, tool_call_id=call_id, name=name)
        self.messages.append(m)
        return m

    # --- reading ----------------------------------------------------

    @property
    def system_prompt(self) -> str | None:
        for m in self.messages:
            if m.role == "system":
                return m.content
        return None

    def to_wire(self) -> list[dict[str, Any]]:
        return [m.to_wire() for m in self.messages]

    def last(self, role: Role | None = None) -> Message | None:
        for m in reversed(self.messages):
            if role is None or m.role == role:
                return m
        return None

    def turns(self) -> list[list[Message]]:
        """Group into turns: a user message and everything that followed it
        until the next user message. The system prompt isn't a turn.

        context.py drops whole turns rather than individual messages, because
        dropping an assistant message and leaving its tool results behind
        produces a history that reads like nonsense -- results with nothing
        that asked for them.
        """
        out: list[list[Message]] = []
        cur: list[Message] = []
        for m in self.messages:
            if m.role == "system":
                continue
            if m.role == "user" and cur:
                out.append(cur)
                cur = []
            cur.append(m)
        if cur:
            out.append(cur)
        return out

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        counts: dict[str, int] = {}
        for m in self.messages:
            counts[m.role] = counts.get(m.role, 0) + 1
        inner = " ".join(f"{k}={v}" for k, v in counts.items())
        return f"<Conversation {inner} chars={self.char_total()}>"

    def char_total(self) -> int:
        return sum(m.char_len() for m in self.messages)

    # --- persistence ------------------------------------------------
    # Answers the "what if the process dies" question with something other
    # than a shrug. Deliberately plain JSON so it can be diffed and eyeballed.

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([asdict(m) for m in self.messages], indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Conversation":
        c = cls()
        data = json.loads(Path(path).read_text())
        c.messages = [Message(**d) for d in data]
        return c

    def transcript(self) -> str:
        """Human-readable dump. Used by the CLI's /dump and when I'm trying to
        work out what the model actually saw."""
        lines = []
        for m in self.messages:
            head = m.role.upper()
            if m.name:
                head += f"({m.name})"
            body = m.content if len(m.content) < 500 else m.content[:500] + " ...[cut]"
            lines.append(f"--- {head} ---\n{body}")
        return "\n".join(lines)
