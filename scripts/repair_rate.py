"""How often does the parser have to fix the model's output?

    .venv/bin/python scripts/repair_rate.py

Runs the same ten tasks through the real agent loop against several models and
reports what fraction of tool-emitting completions were NOT clean JSON.

## Why this script exists

It is the project's own check on itself. The parser sits between the model and
every measurement you could take of the model, and it silently improves the
model's apparent output quality. If you compare two models with a repair
ladder in the loop and don't report the repair rate, you are partly measuring
your parser and reporting it as a model difference.

I know that failure mode specifically: on an earlier research project a
headline improvement turned out on audit to be substantially markdown-fence
stripping in the eval harness. This script is what would have caught it.

## The result (2026-07-25, 10 tasks each, ONE run per cell)

Read the direction, not the digits. These counts come from live generation, so
re-running gives different completion totals. Flagged in review because the
table below reads like a fixed measurement and is not one.

    model                temp  completions  w/ calls  repaired    rate
    llama3.1:8b           0.1           21        11         0     0.0%
    llama3.1:8b           0.9           22        12         0     0.0%
    llama3.2:3b           0.1           29        20         0     0.0%
    llama3.2:3b           0.9           21        11         0     0.0%
    qwen2.5-coder:7b      0.1           20        10         7    70.0%
        all seven were: fenced

Which is not what I expected, and is the interesting part.

**Repair rate is a property of the specific model, not of model size.** Both
llama models emitted clean JSON on every single tool call, at both a low and a
high temperature. The 7B code-tuned model wrapped 70% of its tool calls in
```json fences -- because that is what a model fine-tuned on code has been
trained to do with anything that looks like code.

The consequence, stated plainly: benchmark llama3.1:8b against
qwen2.5-coder:7b on tool use with fence-stripping disabled, and qwen scores
about 70% worse on tool-call validity. It is not 70% worse. That entire gap
would be the parser, and it would look exactly like a capability difference.

Temperature had no visible effect on either llama model, which also surprised
me -- I had assumed higher temperature would degrade format adherence. Ten
tasks per cell is too few to say it has *no* effect, only that it's smaller
than the model-to-model difference.

One more caveat worth stating, since it cuts against my own headline: the 0%
cells are weaker evidence than the 70% cell. "0 repairs out of 11 tool calls"
is consistent with a genuinely low rate rather than a zero one, and a single
run cannot tell those apart. Seven fences out of ten is not a sampling
accident. So the defensible claim is "the code-tuned model fences constantly
and the llamas mostly don't", not "llama never fences".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secret_agent.agent import Agent, AgentFailure
from secret_agent.config import Config
from secret_agent.llm import LLMError, OllamaClient
from secret_agent.parsing import STATS
from secret_agent.permissions import default_permissions
from secret_agent.rag import RAG_TOOLS
from secret_agent.tools.fs import FS_TOOLS
from secret_agent.tools.registry import Registry

TASKS = [
    "List the files in the corpus directory.",
    "Search the docs: what is the default retention window for bronze?",
    "Search the docs for the break-glass duration, then tell me the number.",
    "Search the docs: what is the kettle-highmem cost multiplier?",
    "Find every file in corpus/ that mentions Halberd.",
    "Search the docs for the paging threshold on Lantern p99 latency.",
    "Read the first 5 lines of corpus/glossary.md.",
    "Search the docs: what happens to a team that exceeds its compute budget?",
    "Search the docs for the deduplication key, then explain it in one sentence.",
    "List what's in the corpus folder and then read corpus/retention.md lines 1-10.",
]

CELLS = [
    ("llama3.1:8b", 0.1),
    ("llama3.1:8b", 0.9),
    ("llama3.2:3b", 0.1),
    ("llama3.2:3b", 0.9),
    ("qwen2.5-coder:7b", 0.1),
]


def run_cell(model: str, temp: float):
    STATS.reset()
    cfg = Config(model=model, temperature=temp, max_iterations=8)
    client = OllamaClient(cfg)
    failed = 0
    for task in TASKS:
        reg = Registry(
            list(FS_TOOLS) + list(RAG_TOOLS),
            permissions=default_permissions(auto_approve=True),
        )
        try:
            Agent(reg, client, cfg).run(task)
        except (AgentFailure, LLMError):
            failed += 1
    return failed


def main() -> int:
    print(f"{'model':<18}{'temp':>6}{'compl':>7}{'w/calls':>9}"
          f"{'repaired':>10}{'rate':>8}{'failed':>8}")
    print("-" * 66)

    for model, temp in CELLS:
        try:
            failed = run_cell(model, temp)
        except LLMError as e:
            print(f"{model:<18}{temp:>6}  skipped: {e}")
            continue
        with_calls = STATS.completions_with_calls
        rate = 100 * STATS.completions_needing_repair / max(1, with_calls)
        print(f"{model:<18}{temp:>6}{STATS.completions:>7}{with_calls:>9}"
              f"{STATS.completions_needing_repair:>10}{rate:>7.1f}%{failed:>8}")
        if STATS.repairs:
            print(f"    {dict(STATS.repairs)}")

    print("\nIf you compare models with this runtime in the loop, print this table")
    print("next to the result. A 0% cell and a 70% cell did not run the same")
    print("experiment, and the difference will look like model quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
