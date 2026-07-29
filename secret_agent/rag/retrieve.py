"""Retrieval, exposed to the agent as a tool it decides to call.

## Why a tool and not a preprocessing step

The common RAG shape is: take the user's question, embed it, retrieve, staple
the results onto the prompt, then call the model once. That is a pipeline with
a model at the end, not an agent, and it has three specific problems:

  1. It retrieves for every question, including "hello" and "thanks", paying
     latency and burning context on chunks nobody needed.
  2. It retrieves exactly once, on the original wording. When the first search
     comes back empty or wrong, there is no second attempt -- the model has to
     answer from whatever it got.
  3. It can't compose. "What's the retention default, and does the legal hold
     override it?" is two searches. A preprocessing step does one.

Making retrieval a tool moves the decision to the model: whether to search,
what to search for (it reformulates -- and having watched the transcripts, it
reformulates usefully), how many times, and when it has enough. The agent loop
already handles multi-step tool use, so this costs nothing structurally.

The honest cost: the model might not search when it should, and answer from
its parameters instead. On this fictional corpus that shows up immediately as
a wrong answer. On a corpus of real-world facts it would show up as a
plausible answer that happens not to come from your documents, which is worse
because you cannot see it.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from pydantic import BaseModel, Field

from ..config import Config
from ..tools.base import Tool, ToolError
from .chunking import load_corpus
from .embed import Embedder
from .store_numpy import Hit, NumpyStore

_CFG = Config.from_env()

# Built once and shared. Rebuilding per call would re-embed the corpus on
# every search, which at ~4 seconds a go makes the agent unusable.
_index = None
_index_lock = threading.Lock()


def _store_from_env():
    """Pick the live store from SA_VECTOR_STORE (default numpy).

    This is what makes Qdrant a real integration and not only an eval harness:
    `SA_VECTOR_STORE=qdrant secret-agent --rag "..."` routes the agent's actual
    search_docs tool through Qdrant. Unset or 'numpy' keeps the in-process store,
    so the default install needs no Docker.
    """
    kind = os.environ.get("SA_VECTOR_STORE", "numpy").lower()
    if kind == "numpy":
        return NumpyStore()
    if kind == "qdrant":
        from .store_qdrant import QdrantStore
        return QdrantStore()
    if kind == "chroma":
        from .store_chroma import ChromaStore
        return ChromaStore()
    raise ToolError(f"unknown SA_VECTOR_STORE={kind!r}; use numpy, qdrant, or chroma")


def build_index(
    corpus_dir: str | Path | None = None,
    size: int | None = None,
    overlap: int | None = None,
    store=None,
    embedder: Embedder | None = None,
    verbose: bool = False,
):
    """Chunk -> embed -> store. Returns (store, embedder).

    Takes the store as a parameter so the same function builds a numpy index
    or a Chroma one. That's the whole 6a/6b swap: one argument.
    """
    corpus_dir = Path(corpus_dir or os.environ.get("SA_CORPUS", "corpus"))
    size = size or _CFG.chunk_size
    overlap = overlap if overlap is not None else _CFG.chunk_overlap

    chunks = load_corpus(corpus_dir, size=size, overlap=overlap)
    if not chunks:
        raise ToolError(f"no documents found in {corpus_dir}")

    embedder = embedder or Embedder()
    store = store if store is not None else _store_from_env()

    # Embed with_context(), not raw text: the heading carries topic words the
    # body assumes. Must match what eval.py does or the numbers are wrong.
    vectors = embedder.embed([c.with_context() for c in chunks])
    store.add(chunks, vectors)

    if verbose:
        print(f"indexed {len(chunks)} chunks from {corpus_dir} "
              f"(size={size} overlap={overlap}) -> {store.stats()}")
    return store, embedder


def get_index():
    global _index
    with _index_lock:
        if _index is None:
            _index = build_index()
        return _index


def reset_index():
    """Tests and the ablation need a clean one."""
    global _index
    with _index_lock:
        _index = None


def search(query: str, k: int = 4, index=None) -> list[Hit]:
    store, embedder = index or get_index()
    return store.search(embedder.embed_query(query), k=k)


class SearchDocs(Tool):
    name = "search_docs"
    description = """Search the Meridian documentation for passages relevant to a question.
Use this whenever you are asked about Meridian, its subsystems (Driftwood, Kettle,
Lantern, Ledger), or any of its policies. Do not answer from memory -- you have
not seen this documentation before. Search with a specific question, not a keyword."""
    default_policy = "allow"

    class Args(BaseModel):
        query: str = Field(description="a specific question, e.g. 'what is the default bronze retention window?'")
        k: int = Field(default=4, description="how many passages to return, 1-10")

    def run(self, query: str, k: int = 4) -> str:
        k = max(1, min(10, k))
        try:
            hits = search(query, k=k)
        except Exception as e:
            raise ToolError(f"search failed: {type(e).__name__}: {e}")

        if not hits:
            return f"no passages found for {query!r}"

        return format_hits(hits)


def format_hits(hits: list[Hit]) -> str:
    """What the model sees.

    Scores are included on purpose. A model that can see 0.41 across the board
    behaves differently from one handed four passages with no signal -- in the
    transcripts it hedges appropriately instead of confidently answering from
    a weak match. It is a cheap way to pass uncertainty along instead of
    flattening it.
    """
    out = []
    for i, h in enumerate(hits, 1):
        loc = h.chunk.source + (f" > {h.chunk.heading}" if h.chunk.heading else "")
        out.append(f"[{i}] {loc}  (similarity {h.score:.2f})\n{h.chunk.text}")
    return "\n\n".join(out)


RAG_TOOLS = [SearchDocs]
