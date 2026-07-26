"""RAG tests.

Chunking is tested offline. Anything that embeds is marked `live`, because it
needs ollama with nomic-embed-text pulled -- there's no point mocking an
embedding model, the mock would test nothing.
"""

import numpy as np
import pytest

from secret_agent.rag.chunking import Chunk, load_corpus, split_document, split_text
from secret_agent.rag.eval_set import QUERIES, by_overlap
from secret_agent.rag.store_numpy import NumpyStore

# --- chunking (offline) ----------------------------------------------


def test_short_text_is_one_chunk():
    assert split_text("hello world", size=600) == [("hello world", 0)]


def test_empty_text_gives_nothing():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_overlap_larger_than_size_is_refused_not_hung():
    # this makes no forward progress and loops forever. I hung it once.
    with pytest.raises(ValueError, match="smaller than size"):
        split_text("x" * 1000, size=100, overlap=100)


def test_zero_size_is_refused():
    with pytest.raises(ValueError):
        split_text("x" * 100, size=0)


def test_chunks_overlap_by_roughly_the_requested_amount():
    text = "word " * 500
    chunks = split_text(text, size=400, overlap=100)
    assert len(chunks) > 1
    a, b = chunks[0][0], chunks[1][0]
    # the tail of one should appear at the head of the next
    assert any(a[-n:] in b for n in (30, 50, 80))


def test_splitter_prefers_paragraph_breaks_over_cutting_mid_word():
    text = ("First paragraph, reasonably long so it fills some space here. " * 6
            + "\n\n"
            + "Second paragraph, also long enough to matter for the split. " * 6)
    chunks = [c for c, _ in split_text(text, size=420, overlap=40)]
    # no chunk should end mid-word
    for c in chunks[:-1]:
        assert not c.endswith(tuple("abcdefghijklmnopqrstuvwxyz")) or " " in c[-30:]


def test_it_covers_the_whole_document():
    text = "\n\n".join(f"Paragraph {i} with some content in it." for i in range(40))
    chunks = [c for c, _ in split_text(text, size=300, overlap=50)]
    joined = " ".join(chunks)
    for i in (0, 17, 39):
        assert f"Paragraph {i} " in joined


def test_headings_are_attached_to_chunks(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text(
        "# Title\n\nintro text here\n\n## Section One\n\n"
        + ("body of section one. " * 40)
        + "\n\n## Section Two\n\n"
        + ("body of section two. " * 40)
    )
    chunks = split_document(doc, tmp_path, size=400, overlap=50)
    headings = {c.heading for c in chunks}
    assert "Section One" in headings
    assert "Section Two" in headings


def test_with_context_prepends_source_and_heading():
    c = Chunk(text="14 days", source="retention.md", index=3, heading="Tiers")
    s = c.with_context()
    assert s.startswith("[retention.md > Tiers]")
    assert "14 days" in s


def test_corpus_readme_is_not_indexed():
    # it's a document ABOUT the corpus. Indexing it means queries retrieve
    # the meta-doc, which quietly makes the eval look worse.
    chunks = load_corpus("corpus", 600, 100)
    assert not any(c.source == "README.md" for c in chunks)
    assert len(chunks) > 20


def test_chunk_ids_are_unique():
    chunks = load_corpus("corpus", 600, 100)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


# --- the eval set itself ---------------------------------------------


def test_every_gold_marker_actually_exists_in_its_source_document():
    """The eval set is hand-written, so it can be wrong. If a marker isn't in
    the document, that query is unanswerable and would silently depress every
    score with no indication why."""
    chunks = load_corpus("corpus", 600, 100)
    for q in QUERIES:
        matches = [c for c in chunks if q.is_relevant(c)]
        assert matches, f"no chunk satisfies {q.question!r} (marker={q.marker!r})"


def test_gold_markers_survive_rechunking():
    # the whole reason gold is (source, marker) and not a chunk index
    for size in (200, 600, 2000):
        chunks = load_corpus("corpus", size, size // 6)
        for q in QUERIES:
            assert any(q.is_relevant(c) for c in chunks), \
                f"{q.question!r} has no gold chunk at size={size}"


def test_the_eval_set_has_both_easy_and_honest_questions():
    assert len(QUERIES) >= 15
    assert len(by_overlap("low")) >= 5, "need paraphrased questions or the score is flattered"
    assert len(by_overlap("high")) >= 5


# --- numpy store (offline, synthetic vectors) ------------------------


def _chunk(i):
    return Chunk(text=f"chunk {i}", source="x.md", index=i)


def test_store_ranks_by_cosine_similarity():
    s = NumpyStore()
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    s.add([_chunk(i) for i in range(3)], vecs)

    hits = s.search(np.array([1.0, 0.0], dtype=np.float32), k=3)
    assert [h.chunk.index for h in hits] == [0, 2, 1]
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_empty_store_returns_nothing_rather_than_erroring():
    assert NumpyStore().search(np.array([1.0, 0.0], dtype=np.float32)) == []


def test_k_larger_than_the_corpus_is_clamped():
    s = NumpyStore()
    s.add([_chunk(0)], np.array([[1.0, 0.0]], dtype=np.float32))
    assert len(s.search(np.array([1.0, 0.0], dtype=np.float32), k=50)) == 1


def test_mismatched_lengths_are_caught():
    s = NumpyStore()
    with pytest.raises(ValueError, match="chunks but"):
        s.add([_chunk(0), _chunk(1)], np.array([[1.0, 0.0]], dtype=np.float32))


def test_adding_twice_appends():
    s = NumpyStore()
    v = np.array([[1.0, 0.0]], dtype=np.float32)
    s.add([_chunk(0)], v)
    s.add([_chunk(1)], v)
    assert len(s) == 2


# --- live: embeddings and end-to-end ---------------------------------


@pytest.fixture(scope="module")
def index():
    from secret_agent.rag.embed import Embedder
    chunks = load_corpus("corpus", 600, 100)
    emb = Embedder()
    store = NumpyStore()
    store.add(chunks, emb.embed([c.with_context() for c in chunks]))
    return chunks, store, emb


@pytest.mark.live
def test_embeddings_are_768d_and_normalised(index):
    _, store, _ = index
    assert store.vectors.shape[1] == 768
    norms = np.linalg.norm(store.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


@pytest.mark.live
def test_query_and_document_prefixes_produce_different_vectors(index):
    from secret_agent.rag.embed import DOC_PREFIX, QUERY_PREFIX
    _, _, emb = index
    text = "how long does break-glass access last?"
    as_doc = emb.embed([text], prefix=DOC_PREFIX)[0]
    as_query = emb.embed([text], prefix=QUERY_PREFIX)[0]
    assert not np.allclose(as_doc, as_query)


@pytest.mark.live
def test_retrieval_finds_the_right_document(index):
    from secret_agent.rag.retrieve import search
    hits = search("what is the default retention window for bronze?", k=3, index=index[1:])
    assert hits[0].chunk.source == "retention.md"
    assert any("14 days" in h.chunk.text for h in hits)


@pytest.mark.live
def test_eval_harness_runs_and_reports(index):
    from secret_agent.rag.eval import aggregate, evaluate
    chunks, store, emb = index
    res = evaluate(chunks, store, emb)
    agg = aggregate(res)
    assert len(res) == len(QUERIES)
    # not asserting a specific score -- that's a moving target and pinning it
    # turns an honest measurement into a test that has to be edited to pass.
    # Asserting only that it's better than chance and the metrics are sane.
    assert 0.0 <= agg["hit@3"] <= 1.0
    assert agg["hit@3"] >= agg["hit@1"]
    assert agg["hit@5"] >= agg["hit@3"]
    assert agg["hit@3"] > 0.5, "retrieval is no better than guessing"


@pytest.mark.live
def test_chroma_and_numpy_return_the_same_ranking(index):
    """If the two stores disagree, the interface is leaking."""
    pytest.importorskip("chromadb")
    from secret_agent.rag.store_chroma import ChromaStore

    chunks, numpy_store, emb = index
    chroma = ChromaStore()
    chroma.add(chunks, numpy_store.vectors)

    q = emb.embed_query("how long does break-glass access last?")
    a = [(h.chunk.id, round(h.score, 4)) for h in numpy_store.search(q, k=5)]
    b = [(h.chunk.id, round(h.score, 4)) for h in chroma.search(q, k=5)]
    assert a == b


@pytest.mark.live
def test_chroma_metadata_filter_scopes_the_search(index):
    """The thing numpy can't do without a second index."""
    pytest.importorskip("chromadb")
    from secret_agent.rag.store_chroma import ChromaStore

    chunks, numpy_store, emb = index
    chroma = ChromaStore()
    chroma.add(chunks, numpy_store.vectors)

    q = emb.embed_query("how long does break-glass access last?")
    hits = chroma.search(q, k=3, source="cost-model.md")
    assert hits and all(h.chunk.source == "cost-model.md" for h in hits)


@pytest.mark.live
def test_the_agent_answers_from_the_corpus_not_from_memory():
    """Phase 6 'done when'. MER-4471 is invented; the model cannot know it."""
    from secret_agent.agent import Agent
    from secret_agent.config import Config
    from secret_agent.llm import OllamaClient
    from secret_agent.permissions import default_permissions
    from secret_agent.rag import RAG_TOOLS
    from secret_agent.tools.registry import Registry

    cfg = Config(max_iterations=6)
    reg = Registry(RAG_TOOLS, permissions=default_permissions(auto_approve=True))
    run = Agent(reg, OllamaClient(cfg), cfg).run(
        "What error code does Meridian return when you try to delete a dataset "
        "under legal hold?"
    )
    assert "MER-4471" in run.answer
    assert run.tool_calls >= 1


def test_gold_marker_is_present_in_every_named_source():
    """The methodological fix from the external review, 2026-07-25.

    `test_every_gold_marker_actually_exists_in_its_source_document` only
    checks that SOME chunk satisfies the query. That passed while the
    "why did they stop using the old platform" gold label named
    `deprecations.md` as a source -- and "Halberd" appears nowhere in
    `deprecations.md`. The label was satisfiable by glossary.md alone, so the
    unsound half was invisible.

    This checks each named source individually. A source that cannot contain
    the marker is a labelling error, and it inflates `total_relevant`
    denominators in recall while contributing nothing.
    """
    from pathlib import Path
    for q in QUERIES:
        for src in q.sources:
            text = (Path("corpus") / src).read_text().lower()
            assert q.marker.lower() in text, (
                f"{q.question!r} names {src} as a gold source but "
                f"{q.marker!r} does not appear in it"
            )


def test_gold_markers_are_specific_not_topic_words():
    """A marker must identify the passage that ANSWERS the question.

    The review flagged `marker="operator"` -- a bare word appearing three
    times in access-control.md, so chunks that merely mention the role scored
    as relevant. Cheap proxy for specificity: a marker should not match more
    than half the chunks of its own source document.
    """
    chunks = load_corpus("corpus", 600, 100)
    for q in QUERIES:
        for src in q.sources:
            in_src = [c for c in chunks if c.source == src]
            matching = [c for c in in_src if q.marker.lower() in c.text.lower()]
            assert len(matching) <= max(1, len(in_src) // 2), (
                f"{q.marker!r} matches {len(matching)}/{len(in_src)} chunks of "
                f"{src} -- too generic to mark an answer"
            )
