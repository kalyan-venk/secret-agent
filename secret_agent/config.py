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


@dataclass
class Config:
    # --- model ---
    model: str = "llama3.1:8b"
    host: str = "http://localhost:11434"
    num_ctx: int = DEFAULT_NUM_CTX
    temperature: float = 0.1  # tool-calling wants boring, not creative
    request_timeout: float = 120.0

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

    @classmethod
    def from_env(cls) -> "Config":
        root = Path(os.environ.get("SA_ROOT", os.getcwd())).resolve()
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
        )

    @property
    def history_budget(self) -> int:
        """Tokens available for conversation history, i.e. not the reply."""
        return self.num_ctx - self.response_reserve_tokens
