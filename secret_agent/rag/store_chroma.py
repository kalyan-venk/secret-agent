"""The same store, backed by Chroma.

Written second, on purpose. store_numpy.py is fifteen lines of matrix
multiply; this file is the same interface over a real vector database. Keeping
both is the point -- the diff between them is a much better answer to "do you
understand vector search" than either one alone.

## What Chroma is actually adding

Being specific, because "we use a vector DB" is not an answer:

  persistence      the numpy store lives and dies with the process. Chroma
                   writes a sqlite file plus its index, so a restart doesn't
                   mean re-embedding the corpus.
  ANN index        HNSW rather than exact search. Sub-linear query time, at
                   the cost of *sometimes returning the wrong neighbours* --
                   which is a trade nobody should make at 38 chunks and
                   everybody should make at ten million.
  metadata filters where-clauses alongside the vector query, so you can scope
                   a search to one source document without a second index.
  concurrency      multiple readers/writers without hand-rolling a lock.

At this corpus size it buys none of that in practice. It's here because
"vector database" is a hiring keyword and because the swap being trivial is
itself the evidence that the abstraction was drawn in the right place.

## The distance/similarity trap

Chroma returns a **distance**, numpy returns a **similarity**. They sort in
opposite directions. For normalised vectors Chroma's cosine distance is
`1 - cosine_similarity`, so converting back is a subtraction -- but if you
forget, everything still runs and quietly returns the LEAST relevant chunks
first. I did forget, briefly, and the symptom was an eval score of about zero
with no error anywhere.

Also: Chroma will happily embed for you with its own default model. Passing
embeddings in explicitly is deliberate -- otherwise the two stores would be
using different embedding models and any comparison between them would be
measuring that instead of the store.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .chunking import Chunk
from .store_numpy import Hit

try:
    import chromadb
    from chromadb.config import Settings
    HAVE_CHROMA = True
except ImportError:  # pragma: no cover
    HAVE_CHROMA = False


class ChromaStore:
    """Same interface as NumpyStore: add(chunks, vectors), search(vec, k)."""

    name = "chroma"

    def __init__(self, path: str | Path | None = None, collection: str = "meridian",
                 reset: bool = True):
        if not HAVE_CHROMA:
            raise ImportError(
                "chromadb isn't installed. `pip install -e '.[vector]'`, or "
                "use NumpyStore -- they're interchangeable."
            )

        if path is None:
            self.client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(path), settings=Settings(anonymized_telemetry=False)
            )

        if reset:
            try:
                self.client.delete_collection(collection)
            except Exception:
                pass  # didn't exist; chroma's exception type moves between versions

        self.collection = self.client.get_or_create_collection(
            name=collection,
            # cosine, to match NumpyStore. The default is l2, and on
            # normalised vectors l2 ranks the same way -- but only on
            # NORMALISED vectors, which is a footgun waiting for the day
            # someone passes raw embeddings.
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return

        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=[v.tolist() for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "heading": c.heading, "index": c.index}
                for c in chunks
            ],
        )
        for c in chunks:
            self._chunks[c.id] = c

    def search(self, query_vector: np.ndarray, k: int = 4, source: str | None = None):
        if not self._chunks:
            return []

        where = {"source": source} if source else None
        res = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(k, len(self._chunks)),
            where=where,
        )

        ids = res["ids"][0]
        dists = res["distances"][0]

        # distance -> similarity. See the module docstring; forgetting this
        # inverts the ranking and nothing errors.
        return [
            Hit(chunk=self._chunks[i], score=1.0 - float(d))
            for i, d in zip(ids, dists)
        ]

    def __len__(self) -> int:
        return len(self._chunks)

    def stats(self) -> str:
        return f"{len(self._chunks)} chunks in chroma (hnsw, cosine)"
