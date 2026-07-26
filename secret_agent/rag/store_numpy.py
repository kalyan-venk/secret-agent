"""A vector store that is a numpy array and a list.

Written before reaching for Chroma, deliberately. The point of doing it this
way first is that a vector database stops being magic: it is a matrix of
normalised vectors, a dot product, and an argsort. Everything a real one adds
-- persistence, an ANN index, metadata filtering, concurrent writes -- is a
scaling concern layered on top of these fifteen lines, not a different idea.

Exact search, no approximation. At 60-ish chunks that's a 60x768 matmul,
which is microseconds. ANN indexes (HNSW, IVF) exist because exact search is
O(n) in the corpus size and stops being free somewhere around a million
vectors -- and they buy that speed by sometimes returning the wrong
neighbours. At this scale paying for an approximation would be paying a
correctness cost for a speed problem I don't have.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float

    def __str__(self) -> str:
        return f"{self.chunk.id} ({self.score:.3f})"


class NumpyStore:
    """The reference implementation. store_chroma.py matches this interface."""

    name = "numpy"

    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if self.vectors is None:
            self.vectors = vectors.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[Hit]:
        if self.vectors is None or not len(self.chunks):
            return []

        # Cosine similarity. Both sides are already L2-normalised at embed
        # time, so the dot product IS the cosine -- no division needed.
        scores = self.vectors @ query_vector.astype(np.float32)

        k = min(k, len(scores))
        # argpartition is O(n) vs argsort's O(n log n). Irrelevant at this
        # size; kept because it's the correct reflex and costs nothing.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [Hit(chunk=self.chunks[i], score=float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self.chunks)

    def stats(self) -> str:
        if self.vectors is None:
            return "empty store"
        n, d = self.vectors.shape
        kb = self.vectors.nbytes / 1024
        return f"{n} chunks, {d} dims, {kb:.0f}KB in memory"
