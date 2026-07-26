"""Splitting documents into retrievable pieces.

Chunk size and overlap are decisions, not defaults, so here is the reasoning
rather than two numbers with no history.

## What chunk size trades off against

Both directions lose, and they lose differently:

  Small chunks (~200 chars)
    Retrieval precision goes UP -- the embedding is dominated by one idea, so
    a query about that idea matches sharply. But answer completeness goes
    DOWN, because the retrieved chunk often contains the topic sentence and
    not the number you needed. The classic failure is retrieving "Driftwood
    batches records before landing them." and not the two bullet points
    underneath it that say 50,000 and 90 seconds.

  Large chunks (~2000 chars)
    Answer completeness goes UP -- whatever you retrieve probably contains
    the full answer with its context. But precision goes DOWN, because the
    embedding is now an average of five topics and matches everything
    mediocrely. And you burn context: at top-4 with 2000-char chunks you have
    spent 8000 characters of window on retrieval alone.

600 is where I landed, and the ablation in eval.py is what justifies it
rather than my taste (`--ablate`, 2026-07-25, 20 queries, overlap held at
size/6):

    size  chunks   hit@1   hit@3   hit@5     R@3     MRR
     200     105    0.45    0.65    0.75    0.62   0.586
     300      71    0.55    0.80    0.85    0.72   0.691
     600      38    0.60    0.85    0.90    0.80   0.741   <- configured
    1200      18    0.65    0.75    0.80    0.68   0.727
    2000      10    0.60    0.80    0.95    0.75   0.740

600 is best on hit@3, recall@3 and MRR. The curve has the shape the theory
predicts -- bad at both ends, best in the middle -- which is reassuring, but
note that 1200 wins hit@1 and 2000 wins hit@5. Those are one- and two-query
differences on a 20-query set, so the right reading is "600 is a defensible
middle and the ends are genuinely worse", not "600 is optimal to two decimal
places".

On a corpus of long prose rather than structured documents with tables I
would expect a different answer, and the tuning would need redoing.

## Overlap

100 characters, i.e. ~17%. Overlap exists for one reason: a fact that
straddles a boundary is otherwise in neither chunk in a usable form. The cost
is duplicate content in the index (more storage, and near-duplicate results
crowding out top-k).

## Splitting on structure, not on character count

Splitting blindly at N characters cuts sentences and, worse, cuts tables and
list items in half -- and this corpus is full of both. So the splitter walks
down a preference ladder: paragraph breaks, then line breaks, then sentence
ends, then a hard cut. A chunk that ends mid-word is a last resort, not the
normal case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    text: str
    source: str          # relative path of the document
    index: int           # position within that document
    heading: str = ""    # nearest markdown heading above it
    start: int = 0       # char offset in the original, for debugging

    @property
    def id(self) -> str:
        return f"{self.source}#{self.index}"

    def with_context(self) -> str:
        """What actually goes to the model.

        The heading is prepended because a chunk from the middle of a
        document is frequently unusable without it -- "14 days" means nothing
        until you know you're reading the Tiers table in retention.md. It
        also measurably helps the embedding, since the heading carries the
        topic words the body assumes.
        """
        head = f"[{self.source}"
        if self.heading:
            head += f" > {self.heading}"
        head += "]"
        return f"{head}\n{self.text}"


# preference ladder: try to break at the first of these that's available
_SPLIT_PATTERNS = [
    "\n\n",     # paragraph
    "\n",       # line (matters for tables and lists)
    ". ",       # sentence
    ", ",       # clause -- desperate
    " ",        # word -- more desperate
]

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _headings_by_offset(text: str) -> list[tuple[int, str]]:
    return [(m.start(), m.group(2).strip()) for m in _HEADING.finditer(text)]


def _heading_for(offset: int, headings: list[tuple[int, str]]) -> str:
    current = ""
    for pos, title in headings:
        if pos <= offset:
            current = title
        else:
            break
    return current


def split_text(text: str, size: int = 600, overlap: int = 100) -> list[tuple[str, int]]:
    """Return [(chunk_text, start_offset)].

    Kept separate from split_document so it can be tested without a file.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        # this would make no forward progress and loop forever. Caught here
        # rather than hanging, because I did hang it once.
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [(text, 0)]

    out: list[tuple[str, int]] = []
    pos = 0

    while pos < len(text):
        end = min(pos + size, len(text))

        if end < len(text):
            # walk the ladder looking for a clean break in the last third of
            # the window. Only the last third: allowing a break anywhere lets
            # a paragraph mark near the start produce a 40-char chunk.
            window_start = pos + (2 * size) // 3
            best = -1
            for pat in _SPLIT_PATTERNS:
                found = text.rfind(pat, window_start, end)
                if found > best:
                    best = found + len(pat)
                    break
            if best > pos:
                end = best

        chunk = text[pos:end].strip()
        if chunk:
            out.append((chunk, pos))

        if end >= len(text):
            break
        pos = max(pos + 1, end - overlap)

    return out


def split_document(path: Path, root: Path, size: int = 600, overlap: int = 100) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    headings = _headings_by_offset(text)
    rel = str(path.relative_to(root))

    chunks = []
    for i, (body, offset) in enumerate(split_text(text, size, overlap)):
        chunks.append(
            Chunk(text=body, source=rel, index=i,
                  heading=_heading_for(offset, headings), start=offset)
        )
    return chunks


def load_corpus(root: str | Path, size: int = 600, overlap: int = 100,
                glob: str = "*.md") -> list[Chunk]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")

    chunks: list[Chunk] = []
    for f in sorted(root.rglob(glob)):
        if f.name == "README.md":
            # the corpus README is about the corpus, not part of it. Indexing
            # it means queries retrieve the meta-document, which is a subtle
            # way to make an eval look worse than it is.
            continue
        chunks.extend(split_document(f, root, size, overlap))
    return chunks
