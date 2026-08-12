"""Config for the runtime.

Everything is overridable by env var because I kept wanting to flip the model
without editing a file mid-debug. Prefix is SA_.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        # a typo in an env var shouldn't take the process down silently, but it
        # also shouldn't be ignored silently. Loud and keep going.
        print(f"[config] {name}={raw!r} isn't an int, using {default}")
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _strip_inline_comment(value: str) -> str:
    """Drop a trailing `# comment` from an unquoted .env.local value.

    A line like `KEY=val  # note` used to store the value as
    `"val  # note"` -- the comment silently became part of whatever read the
    key. Only a `#` preceded by whitespace (or right at the start) counts as
    a comment starter; a bare `#` glued to real content (rare, but a token
    could contain one) is left alone rather than guessed at, since a wrong
    guess here would corrupt a real value.
    """
    for i, ch in enumerate(value):
        if ch == "#" and (i == 0 or value[i - 1].isspace()):
            return value[:i].rstrip()
    return value


def _load_dotenv_local(candidates: list[Path] | None = None) -> None:
    """Populate os.environ from an optional .env.local, without overwriting
    anything already set. A real shell export always wins over the file.

    This is a convenience for the hosted provider path (API keys), not a
    required config mechanism -- offline/Ollama use needs no env file at all.
    Deliberately hand-rolled rather than pulling in python-dotenv for a
    four-line KEY=value reader.

    `candidates` is injectable so tests can point this at a tmp_path and get
    a result that depends on nothing else on disk. Real callers (from_env)
    pass nothing and get the default: the repo root (two levels above this
    file) and the current working directory, repo root first so it works
    regardless of where the process was launched from.
    """
    if candidates is None:
        candidates = [
            Path(__file__).resolve().parent.parent / ".env.local",
            Path.cwd() / ".env.local",
        ]
    seen: set[Path] = set()
    for p in candidates:
        if p in seen or not p.exists():
            continue
        seen.add(p)
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if value and value[0] in "\"'":
                # Quoted value: take exactly what's between the matching
                # quotes and drop everything after the closing quote --
                # that's how `KEY="val"  # note` stops "# note" from
                # becoming part of the value, without needing a separate
                # comment-stripping pass to also handle the quoted case.
                quote = value[0]
                end = value.find(quote, 1)
                value = value[1:end] if end != -1 else value[1:]
            else:
                value = _strip_inline_comment(value).strip()
            if key:
                os.environ.setdefault(key, value)


# The one that bit me. `ollama show llama3.1:8b` reports "context length 131072"
# and that number is a lie about what you actually get: the server loads the
# model with its own num_ctx default (small) unless you pass num_ctx in
# options. Nothing errors. The oldest messages just quietly stop existing, and
# you sit there wondering why the model forgot turn 1.
#
# 8192 is what I settled on: big enough that Phase 5 has to do real work on a
# medium conversation, small enough that an 8B model on this laptop still
# answers in a few seconds. Raise it if you have the RAM; KV cache grows
# roughly linearly with this.
DEFAULT_NUM_CTX = 8192

# The model LLM_PROVIDER=vllm requests by default. Must equal the model
# scripts/serve_vllm.sh serves, or vLLM's OpenAI server 404s the call.
VLLM_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class Config:
    # --- model ---
    model: str = "llama3.1:8b"
    host: str = "http://localhost:11434"
    num_ctx: int = DEFAULT_NUM_CTX
    temperature: float = 0.1  # tool-calling wants boring, not creative
    request_timeout: float = 120.0

    # --- provider selection ---
    # "ollama" (default, local, no key), "vllm" (a local vLLM server, the
    # serving backend for the HTTP API under real concurrency), or "hosted"
    # (any OpenAI-compatible endpoint -- Groq, Gemini's OpenAI-compat endpoint,
    # OpenRouter, ...). vllm and hosted share one HostedClient; only the
    # base_url/key defaults differ. See llm.py's build_llm_client().
    llm_provider: str = "ollama"

    # Defaults point at Groq's free tier because it's the provider this was
    # built and tested against. Swapping to Gemini or OpenRouter is base_url
    # + model + key, no code change -- that's the point of the abstraction.
    hosted_base_url: str = "https://api.groq.com/openai/v1"
    hosted_model: str = "llama-3.1-8b-instant"
    # Read from SA_HOSTED_API_KEY if set, else GROQ_API_KEY, so the Groq
    # default needs zero-config beyond exporting GROQ_API_KEY, while a
    # different provider can be pointed at explicitly.
    hosted_api_key: str = ""

    # Tool-call style. "prompted" = schemas go in the system prompt and we parse
    # the model's raw text ourselves. "native" = hand the schemas to Ollama's
    # /api/chat tools field and let it do the structuring.
    #
    # Default is prompted, deliberately. Native works and is one flag away, but
    # it hides every malformed-output case behind Ollama's grammar constraints,
    # and handling those cases is most of what this project is. See DECISIONS.md.
    tool_mode: str = "prompted"

    # --- agent loop ---
    max_iterations: int = 15
    max_parse_retries: int = 2

    # --- context management (phase 5) ---
    # Reserve for the model's own reply. If you budget the whole window for
    # history the model has nowhere to put the answer.
    response_reserve_tokens: int = 1024
    # Below this, don't bother compacting.
    compaction_headroom: float = 0.85
    strategy: str = "summarize"  # or "truncate"
    keep_recent_turns: int = 4
    # A grep can return 4000 lines. Anything longer than this gets cut down
    # before it enters history; the whole thing goes to disk.
    max_tool_result_chars: int = 4000

    # --- rag (phase 6) ---
    embed_model: str = "nomic-embed-text"
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 4

    # --- tools / safety (phase 4) ---
    root: Path = field(default_factory=Path.cwd)
    auto_approve: bool = False  # tests and scripted runs set this

    verbose: bool = False

    # --- LLM call hardening ---
    # 1 initial try + up to 3 retries, jittered exponential backoff, only on
    # a rate limit / timeout / connection error / 5xx. See llm.py.
    llm_max_attempts: int = 4

    # --- per-call telemetry ---
    # Off by default. When on, every LLM call appends one JSON line (model,
    # latency, token counts, retry count, error if any) to call_log_path.
    # Local only -- never sent anywhere. Relative paths resolve against
    # `root`, same as the .spill/ convention in context.py.
    log_calls: bool = False
    call_log_path: Path = field(default_factory=lambda: Path("llm_calls.jsonl"))

    @classmethod
    def from_env(cls) -> "Config":
        _load_dotenv_local()
        root = Path(os.environ.get("SA_ROOT", os.getcwd())).resolve()
        provider = os.environ.get("LLM_PROVIDER", cls.llm_provider).strip().lower()
        # vLLM serves an OpenAI-compatible endpoint, so it rides the same hosted
        # path as Groq; only the defaults change. A vLLM server listens on
        # :8000 by default and needs no key unless started with --api-key, but
        # the openai client still wants a non-empty string, hence "EMPTY". Point
        # the deployed API at vLLM with LLM_PROVIDER=vllm; local dev stays on
        # Ollama, which is what the repair ladder was built against.
        if provider == "vllm":
            hosted_base_url = os.environ.get("SA_HOSTED_BASE_URL", "http://localhost:8000/v1")
            hosted_api_key = os.environ.get("SA_HOSTED_API_KEY") or "EMPTY"
            # Must match what scripts/serve_vllm.sh serves, or vLLM's OpenAI
            # server 404s the request as an unknown model. Both default to this
            # id; test_vllm_default_model_matches_serve_script pins them together.
            hosted_model = os.environ.get("SA_HOSTED_MODEL", VLLM_DEFAULT_MODEL)
        else:
            hosted_base_url = os.environ.get("SA_HOSTED_BASE_URL", cls.hosted_base_url)
            hosted_api_key = (
                os.environ.get("SA_HOSTED_API_KEY")
                or os.environ.get("GROQ_API_KEY")
                or cls.hosted_api_key
            )
            hosted_model = os.environ.get("SA_HOSTED_MODEL", cls.hosted_model)
        return cls(
            model=os.environ.get("SA_MODEL", cls.model),
            host=os.environ.get("SA_HOST", cls.host),
            num_ctx=_env_int("SA_NUM_CTX", DEFAULT_NUM_CTX),
            tool_mode=os.environ.get("SA_TOOL_MODE", cls.tool_mode),
            max_iterations=_env_int("SA_MAX_ITER", cls.max_iterations),
            strategy=os.environ.get("SA_STRATEGY", cls.strategy),
            embed_model=os.environ.get("SA_EMBED_MODEL", cls.embed_model),
            chunk_size=_env_int("SA_CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_env_int("SA_CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_env_int("SA_TOP_K", cls.top_k),
            root=root,
            auto_approve=_env_bool("SA_AUTO_APPROVE", False),
            verbose=_env_bool("SA_VERBOSE", False),
            llm_provider=provider,
            hosted_base_url=hosted_base_url,
            hosted_model=hosted_model,
            hosted_api_key=hosted_api_key,
            llm_max_attempts=_env_int("SA_LLM_MAX_ATTEMPTS", cls.llm_max_attempts),
            log_calls=_env_bool("SA_LOG_CALLS", False),
            call_log_path=Path(
                os.environ.get("SA_CALL_LOG_PATH", "llm_calls.jsonl")
            ),
        )

    @property
    def history_budget(self) -> int:
        return self.num_ctx - self.response_reserve_tokens
