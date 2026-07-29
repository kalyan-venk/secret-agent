"""The same store, backed by a self-hosted Qdrant.

Written third, after store_numpy.py (the reference matrix multiply) and
store_chroma.py (the same interface over an embedded DB). Qdrant is the one a
resume can name: a real vector database, run in Docker, spoken to over its REST
API. The interface is identical to the other two on purpose, so the swap is one
argument and the diff between them is the whole answer to "do you understand
vector search".

## What Qdrant adds over the numpy store

  a server           the numpy store lives and dies with the process. Qdrant is
                     a separate service with its own storage volume, so the
                     embedded corpus survives a restart and a second process can
                     query the same collection.
  an HNSW index      sub-linear query time, at the cost of sometimes returning
                     the wrong neighbours. At 38 chunks that trade buys nothing,
                     and the ANN result matches exact search here. It is the
                     right default at ten million vectors, not at these.
  payload filtering  a where-clause alongside the vector query, so a search can
                     be scoped to one source document without a second index.

At this corpus size it buys none of that in practice. It is here because
"vector database (Qdrant)" is a concrete, defensible anchor and because the
swap being trivial is itself the evidence the abstraction sits in the right
place.

## The distance/similarity trap, the OTHER way round

This is the exact footgun store_chroma.py documents, inverted, and it is worth
being precise because getting it wrong produces no error at all.

  numpy   returns cosine SIMILARITY (dot product of unit vectors). Higher is
          better. No conversion.
  chroma  returns cosine DISTANCE (1 - similarity). Lower is better. You must
          do `1.0 - d` or the ranking comes back worst-first.
  qdrant  with Distance.COSINE returns SIMILARITY in point.score. Higher is
          better, SAME as numpy. So you must NOT invert it. Copying chroma's
          `1.0 - score` here would silently rank the least relevant chunk first
          and the eval would collapse to about zero with nothing raising.

So `score=point.score` verbatim, and there is a parity test in
tests/test_qdrant.py that pins the top-k ordering to NumpyStore precisely so a
future edit cannot reintroduce the inversion unnoticed.

## Embeddings are passed in, never computed here

Qdrant can embed for you via fastembed. It is deliberately not used. The
comparison across stores is only meaningful if every store indexes the SAME
vectors from the SAME model (nomic-embed-text, 768-dim, L2-normalised at embed
time). Letting Qdrant embed would measure fastembed's default model instead of
the store. Same lesson as store_chroma.py.
"""

from __future__ import annotations

import os

import numpy as np

from .chunking import Chunk
from .store_numpy import Hit

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    HAVE_QDRANT = True
except ImportError:  # pragma: no cover
    HAVE_QDRANT = False

# nomic-embed-text is 768-dim. The corpus vectors are L2-normalised at embed
# time, so cosine and dot rank identically; cosine is chosen to match the numpy
# and chroma stores exactly.
VECTOR_SIZE = 768


class QdrantStore:
    """The reference NumpyStore interface, over a self-hosted Qdrant server.

    Defaults connect to the Docker service in docker-compose.yml at
    localhost:6333. Override with SA_QDRANT_URL, or host/port/url args.
    """

    name = "qdrant"

    def __init__(
        self,
        collection: str = "meridian",
        *,
        url: str | None = None,
        host: str | None = None,
        port: int | None = None,
        reset: bool = True,
        vector_size: int = VECTOR_SIZE,
    ):
        if not HAVE_QDRANT:
            raise ImportError(
                "qdrant-client isn't installed. `pip install -e '.[vector]'`, or "
                "use NumpyStore -- they're interchangeable."
            )

        # check_compatibility is off deliberately: the pinned server image and
        # the pip client can sit a few minor versions apart, which trips a noisy
        # warning but not any behaviour we use (upsert + cosine query_points are
        # stable across these versions, and the parity test proves the results
        # match NumpyStore to 1e-4). Pinning both to the same minor is the
        # alternative; this keeps the compose image stable without the noise.
        url = url or os.environ.get("SA_QDRANT_URL")
        if url:
            self.client = QdrantClient(url=url, check_compatibility=False)
        else:
            self.client = QdrantClient(
                host=host or os.environ.get("SA_QDRANT_HOST", "localhost"),
                port=port or int(os.environ.get("SA_QDRANT_PORT", "6333")),
                check_compatibility=False,
            )

        self.collection = collection
        self.vector_size = vector_size
        self._count = 0

        exists = self.client.collection_exists(collection)
        if reset and exists:
            self.client.delete_collection(collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    # COSINE returns SIMILARITY in point.score (higher=better),
                    # matching NumpyStore. Do NOT invert it. See module docstring.
                    distance=qmodels.Distance.COSINE,
                ),
            )
        else:
            self._count = self.client.count(collection, exact=True).count

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return

        points = []
        for i, (c, v) in enumerate(zip(chunks, vectors)):
            # Qdrant point ids must be unsigned int or UUID; chunk.id is a
            # string ("source#index"), so the id is the running position and the
            # human id plus enough to rebuild the Chunk lives in the payload.
            points.append(
                qmodels.PointStruct(
                    id=self._count + i,
                    vector=v.astype(np.float32).tolist(),
                    payload={
                        "chunk_id": c.id,
                        "text": c.text,
                        "source": c.source,
                        "heading": c.heading,
                        "index": c.index,
                    },
                )
            )

        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        self._count += len(points)

    def search(self, query_vector: np.ndarray, k: int = 4, source: str | None = None) -> list[Hit]:
        if self._count == 0:
            return []

        query_filter = None
        if source:
            query_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="source", match=qmodels.MatchValue(value=source)
                )]
            )

        res = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.astype(np.float32).tolist(),
            limit=min(k, self._count),
            query_filter=query_filter,
            with_payload=True,
        ).points

        hits = []
        for point in res:
            p = point.payload or {}
            chunk = Chunk(
                text=p.get("text", ""),
                source=p.get("source", ""),
                index=int(p.get("index", 0)),
                heading=p.get("heading", ""),
            )
            # point.score IS the cosine similarity, higher=better. Verbatim.
            # Inverting it here (1.0 - score) would rank worst-first silently.
            hits.append(Hit(chunk=chunk, score=float(point.score)))
        return hits

    def __len__(self) -> int:
        return self._count

    def stats(self) -> str:
        return f"{self._count} chunks in qdrant (hnsw, cosine, {self.vector_size}d)"
