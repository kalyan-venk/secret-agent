"""Embeddings via Ollama's nomic-embed-text. Local, free, no key.

768 dimensions. Confirmed by asking rather than by reading the model card:

    curl -s localhost:11434/api/embed -d '{"model":"nomic-embed-text","input":"hi"}'

## The asymmetric-prefix thing, and how the measurement went against me

nomic-embed-text is documented as being trained with task prefixes. The same
text embedded as `search_document:` and as `search_query:` produces different
vectors on purpose -- the model is trained so a *question* with the query
prefix lands near the *passage that answers it* with the document prefix.

I wrote a paragraph here asserting that skipping the prefixes is "quietly
worse", which sounded right and which I had not measured. Then I measured it
(`python -m secret_agent.rag.eval --ablate`, 2026-07-25, 20 queries):

    variant                    hit@1   hit@3   hit@5    MRR
    correct (doc/query)         0.60    0.90    0.90   0.756
    none at all                 0.65    0.95    0.95   0.792
    same prefix both sides      0.55    0.95    0.95   0.733

No prefix scored BETTER than the documented-correct usage on every metric.

What I am NOT going to do is present that as a finding. The gap at hit@3 is
18 queries versus 19 -- ONE question, on a 20-question set, over a corpus I
wrote myself. "The model card is wrong" is a large claim to hang on a single
query. Distinguishing a real effect from this would need a few hundred
queries over documents I did not write.

That gap was two queries until an external review found two unsound gold
labels and fixing them moved the baseline up. Worth noting which way that
cuts: the correction made the case for "it's noise" *stronger*, and if I had
already acted on the original result I would now be defending a default
chosen on evidence that had since halved.

So the prefixes stay on, because the documented behaviour is the better prior
when the evidence is this thin. It is logged as an open question in
LEARNING-STATUS.md rather than quietly resolved in whichever direction
happened to score higher on one run.

The reason the ablation is here at all: without it I would have shipped the
original paragraph, which asserted a measurement I had never taken.

## Caching

Embedding the corpus takes a few seconds and gets re-run constantly during
the ablation, which re-chunks and re-embeds for every parameter setting. So
embeddings are cached on disk keyed by (model, prefix, sha1 of text). The
cache is what makes the ablation take seconds instead of minutes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx
import numpy as np

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

_HOST = os.environ.get("SA_HOST", "http://localhost:11434")
_MODEL = os.environ.get("SA_EMBED_MODEL", "nomic-embed-text")
_CACHE_DIR = Path(os.environ.get("SA_CACHE_DIR", ".embed_cache"))

# Ollama will take a list, but a very large one occasionally times out on
# this laptop. 64 is a size that has never done that.
BATCH = 64


class EmbedError(RuntimeError):
    pass


def _key(text: str, model: str) -> str:
    return hashlib.sha1(f"{model}\x00{text}".encode("utf-8")).hexdigest()


class Embedder:
    def __init__(self, model: str = _MODEL, host: str = _HOST, cache: bool = True):
        self.model = model
        self.host = host
        self.cache_dir = _CACHE_DIR if cache else None
        self._mem: dict[str, list[float]] = {}
        self.api_calls = 0
        self.cache_hits = 0
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()

    def _load_disk_cache(self):
        f = self.cache_dir / f"{self.model.replace(':', '_')}.json"
        if f.exists():
            try:
                self._mem = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                # a corrupt cache is not worth dying over; rebuild it
                self._mem = {}

    def _save_disk_cache(self):
        if not self.cache_dir:
            return
        f = self.cache_dir / f"{self.model.replace(':', '_')}.json"
        try:
            f.write_text(json.dumps(self._mem))
        except OSError:
            pass

    # -----------------------------------------------------------------

    def embed(self, texts: list[str], prefix: str = DOC_PREFIX) -> np.ndarray:
        """-> (n, dim) float32, L2-normalised."""
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)

        prefixed = [prefix + t for t in texts]
        missing = [t for t in prefixed if _key(t, self.model) not in self._mem]

        for i in range(0, len(missing), BATCH):
            batch = missing[i:i + BATCH]
            for text, vec in zip(batch, self._call(batch)):
                self._mem[_key(text, self.model)] = vec

        self.cache_hits += len(prefixed) - len(missing)
        if missing:
            self._save_disk_cache()

        arr = np.array([self._mem[_key(t, self.model)] for t in prefixed], dtype=np.float32)
        return l2_normalise(arr)

    def embed_query(self, text: str) -> np.ndarray:
        """One query vector. Note the DIFFERENT prefix -- see the docstring."""
        return self.embed([text], prefix=QUERY_PREFIX)[0]

    def _call(self, texts: list[str]) -> list[list[float]]:
        self.api_calls += 1
        try:
            r = httpx.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=180,
            )
        except httpx.ConnectError as e:
            raise EmbedError(f"can't reach ollama at {self.host}") from e

        if r.status_code != 200:
            body = r.text[:300]
            if "not found" in body:
                raise EmbedError(
                    f"model {self.model!r} isn't pulled. Run: ollama pull {self.model}"
                )
            raise EmbedError(f"embed failed ({r.status_code}): {body}")

        out = r.json().get("embeddings")
        if not out or len(out) != len(texts):
            raise EmbedError(f"expected {len(texts)} embeddings, got {len(out or [])}")
        return out


def l2_normalise(a: np.ndarray) -> np.ndarray:
    """Normalise rows to unit length.

    Doing this once at index time is why the store can use a plain dot
    product for cosine similarity -- for unit vectors, dot product IS cosine.
    That turns the whole search into one matrix multiply.
    """
    if a.ndim == 1:
        n = np.linalg.norm(a)
        return a / n if n else a
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return a / norms
