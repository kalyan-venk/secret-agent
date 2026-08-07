"""Run the retrieval eval and one full agent task through the hosted provider.

    export GROQ_API_KEY=...      # or put it in .env.local, see .env.local.example
    ollama serve                 # still needed -- see "why ollama serve is still required" below
    .venv/bin/python scripts/hosted_eval.py

Written 2026-08-05 for the hosted-provider extension (EXTENSIONS-2026-08-05.md).
I did not have a GROQ_API_KEY in the environment this was built in, so this
script has never produced real hosted numbers -- it's built and the offline
provider-selection tests pass, but it is UNRUN. Run it once a key is set and
paste the output where this docstring says PASTE HERE.

## What this measures, and what it doesn't

1. **The RAG retrieval eval** (`python -m secret_agent.rag.eval --ablate`):
   hit@k, MRR, both ablations. This does NOT change with LLM_PROVIDER --
   retrieval scoring depends only on the embedder (nomic-embed-text, which
   only exists locally in Ollama; there's no free hosted embedding endpoint
   in this build). Running it here reproduces the numbers already in
   README.md, as a check that nothing about the provider change broke
   retrieval. If the numbers differ from README's hit@3 0.90 / MRR 0.756 /
   low-overlap 0.71, something is wrong -- they should be identical.

2. **One full agent task through the hosted chat model** (LLM_PROVIDER=hosted,
   default Groq llama-3.1-8b-instant): the actual thing this extension adds.
   Tool-calling, parsing, retrieval-as-a-tool all run against Groq instead of
   local llama3.1:8b. This is the number that's new.

## Why `ollama serve` is still required even for the hosted run

Embeddings (nomic-embed-text) have no free hosted equivalent wired up here,
so `search_docs` still calls local Ollama for retrieval even when the chat
model is Groq. Only the reasoning/tool-calling loop moves to the hosted
provider. If you see a connection-refused error, that's ollama, not Groq --
the traceback will say which.

## Output

PASTE HERE once run with a real GROQ_API_KEY:

    RAG eval (hit@3 / MRR / low-overlap hit@3): [fill after hosted run with GROQ_API_KEY]
    Hosted agent task -- iterations / tool calls / repair rate: [fill after hosted run with GROQ_API_KEY]
    Hosted agent task -- answer: [fill after hosted run with GROQ_API_KEY]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secret_agent.agent import Agent, AgentFailure
from secret_agent.config import Config
from secret_agent.llm import LLMError, build_llm_client
from secret_agent.permissions import default_permissions
from secret_agent.rag import RAG_TOOLS
from secret_agent.tools.fs import FS_TOOLS
from secret_agent.tools.registry import Registry

TASK = (
    "Search the docs: what error code does Meridian return when you try to "
    "delete a dataset under legal hold, and is it retryable?"
)


def run_rag_eval() -> int:
    print("=" * 70)
    print("1/2 -- RAG retrieval eval (embeddings, still local Ollama)")
    print("=" * 70)
    result = subprocess.run(
        [sys.executable, "-m", "secret_agent.rag.eval", "--ablate"],
        cwd=Path(__file__).resolve().parent.parent,
    )
    return result.returncode


def run_hosted_agent_task() -> int:
    print("\n" + "=" * 70)
    print("2/2 -- one full agent task through the hosted provider")
    print("=" * 70)

    cfg = Config.from_env()
    cfg.llm_provider = "hosted"
    cfg.max_iterations = 8

    try:
        client = build_llm_client(cfg)
    except LLMError as e:
        print(f"\ncouldn't build the hosted client: {e}", file=sys.stderr)
        print(
            "export GROQ_API_KEY, or set SA_HOSTED_API_KEY / SA_HOSTED_BASE_URL / "
            "SA_HOSTED_MODEL for a different OpenAI-compatible provider.",
            file=sys.stderr,
        )
        return 1

    reg = Registry(
        list(FS_TOOLS) + list(RAG_TOOLS),
        permissions=default_permissions(auto_approve=True),
    )
    agent = Agent(reg, client, cfg)

    print(f"\nprovider: {cfg.llm_provider} ({cfg.hosted_base_url}, {cfg.hosted_model})")
    print(f"task: {TASK}\n")

    try:
        run = agent.run(TASK)
    except AgentFailure as e:
        print(f"agent task FAILED: {e}", file=sys.stderr)
        return 1
    except LLMError as e:
        print(f"hosted call FAILED: {e}", file=sys.stderr)
        return 1

    print(run.answer)
    print(f"\n[{run.summary()}]")
    return 0


def main() -> int:
    rc1 = run_rag_eval()
    rc2 = run_hosted_agent_task()
    return rc1 or rc2


if __name__ == "__main__":
    raise SystemExit(main())
