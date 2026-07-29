"""QdrantStore parity and roundtrip, against a live self-hosted Qdrant.

Marked `qdrant` so it stays out of the default offline suite (it needs the
Docker service on :6333 and qdrant-client installed). Run it with:

    docker compose up -d qdrant
    .venv/bin/pip install -e '.[vector]'
    .venv/bin/pytest -m qdrant -o addopts=""

The sharp claim here is the one the store_qdrant docstring warns about: Qdrant's
COSINE score is a SIMILARITY (higher=better, same as NumpyStore), NOT a distance
needing inversion. If a future edit copied Chroma's `1.0 - score` in, the ranking
would silently flip and the eval would collapse to ~0 with nothing raising. The
parity test pins QdrantStore's top-k ordering to NumpyStore exactly, so that
inversion cannot be reintroduced unnoticed.

Skips gracefully (never fails) if qdrant-client is missing or :6333 is
unreachable, matching the mcp integration-test pattern in test_mcp.py.
"""

import numpy as np
import pytest

from secret_agent.rag.chunking import load_corpus
from secret_agent.rag.embed import DOC_PREFIX, Embedder
from secret_agent.rag.eval_set import QUERIES
from secret_agent.rag.store_numpy import NumpyStore

pytestmark = pytest.mark.qdrant


def _qdrant_up() -> bool:
    try:
        import httpx
        return httpx.get("http://localhost:6333/readyz", timeout=2).status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def corpus_vectors():
    if not _qdrant_up():
        pytest.skip("Qdrant not reachable on :6333 (docker compose up -d qdrant)")
    try:
        import qdrant_client  # noqa: F401
    except ImportError:
        pytest.skip("qdrant-client not installed (pip install -e '.[vector]')")

    chunks = load_corpus("corpus", size=600, overlap=100)
    embedder = Embedder()
    vectors = embedder.embed([c.with_context() for c in chunks], prefix=DOC_PREFIX)
    return chunks, vectors, embedder


@pytest.fixture
def stores(corpus_vectors):
    from secret_agent.rag.store_qdrant import QdrantStore

    chunks, vectors, embedder = corpus_vectors
    numpy_store = NumpyStore()
    numpy_store.add(chunks, vectors)
    qdrant_store = QdrantStore(collection="meridian_test", reset=True)
    qdrant_store.add(chunks, vectors)
    return numpy_store, qdrant_store, embedder


def test_add_reports_the_full_corpus(stores):
    _, qdrant_store, _ = stores
    assert len(qdrant_store) == 38
    assert "38 chunks in qdrant" in qdrant_store.stats()


def test_topk_ordering_matches_numpy_exactly(stores):
    # The inversion trap. If point.score were treated as a distance, this order
    # would reverse. It does not, so the two stores must agree query for query.
    numpy_store, qdrant_store, embedder = stores
    for q in QUERIES:
        qv = embedder.embed_query(q.question)
        n_ids = [h.chunk.id for h in numpy_store.search(qv, k=5)]
        q_ids = [h.chunk.id for h in qdrant_store.search(qv, k=5)]
        assert n_ids == q_ids, f"ordering diverged on: {q.question!r}"


def test_scores_are_similarities_higher_is_better(stores):
    # A similarity is in [-1, 1] and the top hit must not score below the
    # second. If it were an un-inverted distance, the top score would be the
    # SMALLEST, and this would fail.
    _, qdrant_store, embedder = stores
    qv = embedder.embed_query(QUERIES[0].question)
    hits = qdrant_store.search(qv, k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] <= 1.0 + 1e-6
    assert scores[0] >= scores[-1]


def test_scores_match_numpy_similarity(stores):
    numpy_store, qdrant_store, embedder = stores
    qv = embedder.embed_query(QUERIES[0].question)
    n = [h.score for h in numpy_store.search(qv, k=5)]
    q = [h.score for h in qdrant_store.search(qv, k=5)]
    assert np.allclose(n, q, atol=1e-4)


def test_roundtrip_reconnect_sees_persisted_points(corpus_vectors):
    # A fresh client with reset=False must find the collection an earlier writer
    # left behind: the whole point of a server-backed store over the in-process
    # one. Write, then reconnect without resetting and query.
    from secret_agent.rag.store_qdrant import QdrantStore

    chunks, vectors, embedder = corpus_vectors
    writer = QdrantStore(collection="meridian_roundtrip", reset=True)
    writer.add(chunks, vectors)

    reader = QdrantStore(collection="meridian_roundtrip", reset=False)
    assert len(reader) == 38
    hits = reader.search(embedder.embed_query(QUERIES[0].question), k=3)
    assert hits and hits[0].chunk.text
