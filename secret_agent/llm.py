"""Model client.

One interface, two implementations: Ollama (local, default) and an
OpenAI-compatible hosted path (Groq's free tier by default; Gemini's
OpenAI-compat endpoint or OpenRouter work by changing base_url + model + key,
no code change). `build_llm_client` picks one from config so callers don't
have to.

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

try:
    from openai import OpenAI as _OpenAI
except ImportError:  # optional extra ("hosted"); base install stays lean
    _OpenAI = None


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
    # Which Message.to_wire() shape this client needs. See conversation.py.
    WIRE_FORMAT = "ollama"

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.from_env()
        self._client = httpx.Client(
            base_url=self.cfg.host,
            timeout=self.cfg.request_timeout,
        )
        # cheap counter, mostly so I can tell whether the summarizer is
        # firing more often than I think it is
        self.calls = 0


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


class HostedClient:
    """Any OpenAI-compatible endpoint: chat/completions with the same
    request and response shape as the OpenAI API. Groq is the default
    (free tier, no credit card), because that's what this was built and
    tested against, but nothing here is Groq-specific -- point
    `hosted_base_url` at Gemini's OpenAI-compat endpoint or OpenRouter and
    set the matching key and model, and this class doesn't change.

    Deliberately mirrors OllamaClient's shape (same complete() signature,
    same Completion/Usage return, same LLMError on failure) so agent.py and
    cli.py don't need to know which provider they're holding.
    """

    # OpenAI's stricter wire shape: role=tool needs tool_call_id, and the
    # preceding assistant message needs a matching "tool_calls" array. See
    # Message.to_wire() in conversation.py -- that's where this is read.
    WIRE_FORMAT = "hosted"

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.from_env()
        if not self.cfg.hosted_api_key:
            raise LLMError(
                "LLM_PROVIDER=hosted but no API key is set. Export "
                "GROQ_API_KEY (or SA_HOSTED_API_KEY for a different "
                "provider), or put it in .env.local."
            )
        if _OpenAI is None:
            raise LLMError(
                "the `openai` package isn't installed. "
                "pip install -e '.[hosted]'"
            )
        self._client = _OpenAI(
            api_key=self.cfg.hosted_api_key,
            base_url=self.cfg.hosted_base_url,
            timeout=self.cfg.request_timeout,
        )
        self.calls = 0

    def complete(self, messages, tools=None) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.cfg.hosted_model,
            "messages": messages,
            "temperature": self.cfg.temperature,
        }
        # Same rule as OllamaClient: only sent in native mode. In prompted
        # mode the schemas are already text in the system prompt.
        if tools and self.cfg.tool_mode == "native":
            kwargs["tools"] = tools

        self.calls += 1
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # The openai SDK raises its own hierarchy (APIConnectionError,
            # AuthenticationError, RateLimitError, APITimeoutError, ...).
            # Wrapped uniformly so every caller only ever handles LLMError,
            # same as the Ollama path -- they shouldn't need to know or
            # catch a second exception family depending on which provider
            # is configured.
            raise LLMError(
                f"hosted provider call failed ({self.cfg.hosted_base_url}, "
                f"model={self.cfg.hosted_model}): {e}"
            ) from e

        choice = resp.choices[0]
        msg = choice.message

        native_calls: list[dict[str, Any]] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            native_calls.append(
                {
                    # Groq's real id (e.g. "call_abc123"). Used by
                    # parse_native_tool_calls to set ToolCall.id, which then
                    # has to reappear as the following tool message's
                    # tool_call_id -- and as an entry in the assistant
                    # message's own "tool_calls" array when this gets
                    # replayed into history. Dropping this and generating a
                    # fresh id locally (the bug this comment replaced) broke
                    # that pairing: Groq tolerated it, real OpenAI would not.
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        # Left as the SDK's raw string, same as Ollama
                        # sometimes double-encodes -- parse_native_tool_calls
                        # already handles a string here.
                        "arguments": tc.function.arguments,
                    }
                }
            )

        usage = resp.usage
        return Completion(
            text=msg.content or "",
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
            usage=Usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            native_tool_calls=native_calls,
        )

    def count_tokens(self, text: str) -> int:
        # No cheap tokenize-only endpoint on the hosted side (unlike
        # Ollama's num_predict=0 trick). chars/4 is the same fallback
        # OllamaClient uses when its own round trip fails.
        return approx_tokens(text)

    def close(self):
        close_fn = getattr(self._client, "close", None)
        if close_fn:
            close_fn()


def build_llm_client(cfg: Config | None = None) -> LLMClient:
    """Pick a provider from config. This is the one place that decides --
    agent.py and cli.py call this instead of naming OllamaClient directly,
    so LLM_PROVIDER actually has an effect end to end.

    Default (LLM_PROVIDER unset or "ollama") returns exactly what
    OllamaClient() always returned: nothing about the offline path changes.
    """
    cfg = cfg or Config.from_env()
    provider = (cfg.llm_provider or "ollama").strip().lower()
    if provider == "ollama":
        return OllamaClient(cfg)
    if provider in ("hosted", "groq", "openai", "openai-compatible"):
        return HostedClient(cfg)
    raise LLMError(
        f"unknown LLM_PROVIDER={provider!r}. Use 'ollama' (default) or "
        "'hosted' (any OpenAI-compatible endpoint)."
    )


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

    WIRE_FORMAT = "ollama"

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
    try:
        c = build_llm_client(cfg)
        out = c.complete([{"role": "user", "content": prompt}])
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    model_id = cfg.hosted_model if cfg.llm_provider != "ollama" else cfg.model
    print(out.text.strip())
    print(
        f"\n[{cfg.llm_provider}:{model_id} | prompt={out.usage.prompt_tokens} "
        f"completion={out.usage.completion_tokens} num_ctx={cfg.num_ctx}]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
