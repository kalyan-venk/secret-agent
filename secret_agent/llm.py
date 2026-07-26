"""Model client.

One interface, one implementation (Ollama). The interface exists so that
swapping to Anthropic later is a new file rather than a rewrite -- see the
stub at the bottom of DECISIONS.md for what that would look like.

    client.complete(messages, tools=None) -> Completion

That's it. Everything else in this repo talks to the model through that.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .config import Config


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Completion:
    """What came back. `text` is the useful bit; `raw` is kept so I can go
    look at what the server actually said when parsing goes sideways, which
    it does."""

    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    # Only populated in native tool mode. In prompted mode the tool calls are
    # still sitting inside `text` and parsing.py digs them out.
    native_tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.text.strip()) or bool(self.native_tool_calls)


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion: ...


class OllamaClient:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.from_env()
        self._client = httpx.Client(
            base_url=self.cfg.host,
            timeout=self.cfg.request_timeout,
        )
        # cheap counter, mostly so I can tell whether the summarizer is
        # firing more often than I think it is
        self.calls = 0

    # ------------------------------------------------------------------

    def complete(self, messages, tools=None) -> Completion:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {
                # num_ctx MUST be set here. See the long comment in config.py.
                # Leaving it off gets you silent truncation of old messages.
                "num_ctx": self.cfg.num_ctx,
                "temperature": self.cfg.temperature,
            },
        }

        if tools and self.cfg.tool_mode == "native":
            payload["tools"] = tools

        self.calls += 1
        try:
            r = self._client.post("/api/chat", json=payload)
        except httpx.ConnectError as e:
            raise LLMError(
                f"can't reach ollama at {self.cfg.host} -- is `ollama serve` running?"
            ) from e
        except httpx.ReadTimeout as e:
            raise LLMError(
                f"ollama timed out after {self.cfg.request_timeout}s. "
                "8B models on CPU are slow with a big num_ctx; raise SA_TIMEOUT or drop num_ctx."
            ) from e

        if r.status_code != 200:
            raise LLMError(f"ollama returned {r.status_code}: {r.text[:400]}")

        body = r.json()
        msg = body.get("message") or {}

        return Completion(
            text=msg.get("content", "") or "",
            raw=body,
            usage=Usage(
                prompt_tokens=body.get("prompt_eval_count", 0) or 0,
                completion_tokens=body.get("eval_count", 0) or 0,
            ),
            native_tool_calls=msg.get("tool_calls", []) or [],
        )

    def count_tokens(self, text: str) -> int:
        """Ask the server to tokenize without generating.

        This is the honest way to count for a local model -- it's the model's
        own tokenizer, not an approximation. Costs a round trip, so context.py
        only calls it when it actually needs a real number and uses a cheap
        heuristic the rest of the time.
        """
        try:
            r = self._client.post(
                "/api/chat",
                json={
                    "model": self.cfg.model,
                    "messages": [{"role": "user", "content": text}],
                    "stream": False,
                    "options": {"num_ctx": self.cfg.num_ctx, "num_predict": 0},
                },
            )
            r.raise_for_status()
            # prompt_eval_count includes the chat template wrapper (BOS, role
            # headers, etc), so this reads a bit high for short strings. Fine
            # for budgeting -- overcounting is the safe direction.
            return int(r.json().get("prompt_eval_count", 0))
        except Exception:
            return approx_tokens(text)

    def close(self):
        self._client.close()


def approx_tokens(text: str) -> int:
    """chars/4. It's the standard rule of thumb and it's wrong, but it's wrong
    cheaply and in a predictable direction for English prose.

    Measured against llama3.1's real tokenizer on this repo's own README it
    came out ~8% low. On JSON blobs it's worse (~20% low) because punctuation
    and quotes tokenize badly. context.py pads for that.
    """
    return max(1, len(text) // 4)


class EchoClient:
    """Test double. Hands back canned strings in order.

    Exists because I got tired of waiting 6 seconds per turn to test whether
    the loop's iteration cap worked.
    """

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.seen: list[list[dict]] = []
        self.calls = 0

    def complete(self, messages, tools=None) -> Completion:
        self.calls += 1
        self.seen.append(list(messages))
        if not self.responses:
            return Completion(text="(echo client out of responses)")
        return Completion(text=self.responses.pop(0))

    def count_tokens(self, text: str) -> int:
        return approx_tokens(text)


def _main(argv: list[str]) -> int:
    prompt = " ".join(argv[1:]) or "say hi in five words"
    cfg = Config.from_env()
    c = OllamaClient(cfg)
    try:
        out = c.complete([{"role": "user", "content": prompt}])
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(out.text.strip())
    print(
        f"\n[{cfg.model} | prompt={out.usage.prompt_tokens} "
        f"completion={out.usage.completion_tokens} num_ctx={cfg.num_ctx}]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
