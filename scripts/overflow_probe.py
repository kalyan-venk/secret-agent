"""What does Ollama actually do when the prompt exceeds num_ctx?

This is the script that produced the table in context.py's docstring. I wrote
the docstring first from what I assumed ("drops from the front"), then ran
this and found out it was wrong, which is the whole argument for running it.

    .venv/bin/python scripts/overflow_probe.py

Method: put a distinct marker in the system prompt, the first user message and
the last one, pad the middle so the history is far larger than the window, and
ask the model to read the markers back. Whichever markers it can still see are
the ones that survived truncation.

Reading the results: prompt_eval_count is the server telling you how many
tokens it actually processed. If that number is far below the size of what you
sent, the difference is what it threw away without mentioning.
"""

from __future__ import annotations

import sys

import httpx

HOST = "http://localhost:11434"
MODEL = "llama3.1:8b"

FILLER = "Nothing of consequence happened. " * 700

MESSAGES = [
    {"role": "system", "content": "You are terse. SYSTEM_TOKEN=ALPHA."},
    {"role": "user", "content": "Remember: FIRST_TOKEN=BRAVO."},
    {"role": "assistant", "content": "Noted."},
    {"role": "user", "content": FILLER},
    {"role": "assistant", "content": "Understood."},
    {
        "role": "user",
        "content": "Remember: LAST_TOKEN=DELTA. "
                   "Now list every TOKEN value you can see, verbatim.",
    },
]

MARKERS = [("system", "ALPHA"), ("first user", "BRAVO"), ("last user", "DELTA")]


def main() -> int:
    print(f"model: {MODEL}\n")
    print(f"{'num_ctx':>8}{'http':>6}{'prompt_eval':>13}{'error':>8}", end="")
    for label, _ in MARKERS:
        print(f"{label:>12}", end="")
    print()
    print("-" * 74)

    for ctx in (8192, 1024, 256):
        try:
            r = httpx.post(
                f"{HOST}/api/chat",
                json={
                    "model": MODEL,
                    "messages": MESSAGES,
                    "stream": False,
                    "options": {"num_ctx": ctx, "temperature": 0},
                },
                timeout=300,
            )
        except httpx.ConnectError:
            print("ollama isn't running", file=sys.stderr)
            return 1

        body = r.json()
        text = body.get("message", {}).get("content", "")
        print(f"{ctx:>8}{r.status_code:>6}{body.get('prompt_eval_count', -1):>13}"
              f"{body.get('error', 'none'):>8}", end="")
        for _, marker in MARKERS:
            print(f"{'kept' if marker in text else 'DROPPED':>12}", end="")
        print()

    print(
        "\nEvery row is HTTP 200 with no error field. Ollama keeps the system\n"
        "prompt and the most recent message and silently discards the middle.\n"
        "The system prompt surviving is what makes this hard to notice -- the\n"
        "model still sounds like itself, it has just forgotten the task."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
