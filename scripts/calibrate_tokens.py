"""Measure how wrong the cheap token estimate is.

Produces the two constants in context.py (PER_MESSAGE_OVERHEAD and the
punctuation factor in approx()). Run it again if you change models -- the
numbers are per-tokenizer and there is no reason to expect llama3.1's to
match anything else's.

    .venv/bin/python scripts/calibrate_tokens.py

Method: ask the server to tokenize a string with num_predict=0 and read
prompt_eval_count. That includes the chat template wrapper, which is what
PER_MESSAGE_OVERHEAD is measuring -- send the empty string and whatever comes
back is pure wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secret_agent.config import Config
from secret_agent.llm import OllamaClient

PROSE = """The API is stateless. There is no server-side session. Every turn
re-sends the entire history, which is why context management exists at all:
the thing that grows is the payload you send, not some remote object. A small
local model emits messy tool calls and building against it forces the real
parse, validate and retry logic that a frontier model would have hidden."""

JSON_BLOB = """{"name": "read_file", "arguments": {"path": "src/main.py", "start_line": 1, "end_line": 40}}
{"name": "grep", "arguments": {"pattern": "def [a-z_]+\\\\(", "path": ".", "glob": "*.py"}}
{"name": "write_file", "arguments": {"path": "out.json", "content": "{\\"a\\": [1, 2, 3]}"}}"""

CODE = '''def safe_resolve(raw: str, root: Path, *, must_exist: bool = False) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise PathEscape("path must be a non-empty string")
    resolved = (root / Path(raw)).resolve(strict=False)
    if not (resolved == root or resolved.is_relative_to(root)):
        raise PathEscape(f"{raw!r} resolves outside the project root")
    return resolved'''

TOOL_OUTPUT = "\n".join(
    f"src/module_{i}.py:{i * 3}: TODO: this needs handling before release"
    for i in range(30)
)

SAMPLES = [
    ("prose", PROSE),
    ("json", JSON_BLOB),
    ("code", CODE),
    ("tool output", TOOL_OUTPUT),
]


def main() -> int:
    cfg = Config.from_env()
    client = OllamaClient(cfg)

    print(f"model: {cfg.model}\n")

    # --- per-message overhead ---
    empty = client.count_tokens("")
    print(f"chat template overhead on an empty message: {empty} tokens")
    print("  -> PER_MESSAGE_OVERHEAD\n")

    print(f"{'sample':<14}{'chars':>7}{'punct%':>8}{'real':>7}{'chars/4':>9}"
          f"{'err':>8}{'needed':>9}")
    print("-" * 62)

    factors = []
    for label, text in SAMPLES:
        real = client.count_tokens(text) - empty  # strip the wrapper
        naive = len(text) // 4
        punct = sum(1 for c in text if c in '{}[]":,\\') / len(text)
        err = 100 * (naive - real) / real
        needed = len(text) / real  # the chars-per-token that WOULD be right
        factors.append((label, punct, needed))
        print(f"{label:<14}{len(text):>7}{100 * punct:>7.1f}%{real:>7}{naive:>9}"
              f"{err:>7.1f}%{needed:>9.2f}")

    print()
    low = [n for _, p, n in factors if p < 0.05]
    high = [n for _, p, n in factors if p >= 0.05]
    if low:
        print(f"low-punctuation text  (<5%): {sum(low) / len(low):.2f} chars/token")
    if high:
        print(f"high-punctuation text (>=5%): {sum(high) / len(high):.2f} chars/token")
    print("\nThose two numbers are what approx() should use. Negative err means")
    print("chars/4 UNDERcounts, which is the dangerous direction for a budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
