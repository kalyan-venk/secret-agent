"""Does retrieval actually work?

    .venv/bin/python -m secret_agent.rag.eval
    .venv/bin/python -m secret_agent.rag.eval --ablate

This module is the reason the project can claim RAG rather than merely
contain RAG. Without it, "I added retrieval" is a statement about code that
exists, not about whether it helped. An earlier project of mine taught me the
difference the expensive way, when an audit traced most of a headline gain
back to the harness. Being more careful is not the fix. Measuring is, along
with writing down what the measurement does not cover.

## The three metrics, and which one matters

  precision@k  what fraction of the k returned passages are relevant.
               Bounded above by (relevant_chunks / k). With chunk overlap a
               fact often lives in exactly 1-2 chunks, so precision@5 has a
               ceiling near 0.2-0.4 no matter how good retrieval is. Reported
               because it's standard; do not read it as "60% wrong".

  recall@k     what fraction of all relevant passages are in the top k.
               The meaningful denominator.

  hit@k        did at least ONE relevant passage make it into the top k.
               **This is the one that predicts whether the agent can answer
               the question**, because the model needs the fact once, not
               every copy of it. If you quote one number from this harness,
               quote hit@k and say which k.

## What this does not measure

  - Whether the model's final answer is correct. Retrieval can succeed and
    the model still misread the passage. That's a generation eval, and it is
    not this.
  - Anything about a corpus larger than eight documents, or one I did not
    write myself. See corpus/README.md.
  - Multi-hop questions, mostly. Only "why did they stop using the old
    platform" needs two documents.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

from .chunking import load_corpus
from .embed import DOC_PREFIX, QUERY_PREFIX, Embedder
from .eval_set import QUERIES, EvalQuery
from .store_numpy import NumpyStore

KS = (1, 3, 5)


@dataclass
class QueryResult:
    query: EvalQuery
    total_relevant: int
    ranks: list[int]        # 0-indexed positions of relevant hits in the ranking
    top_source: str
    top_score: float

    def hit_at(self, k: int) -> bool:
        return any(r < k for r in self.ranks)

    def precision_at(self, k: int) -> float:
        return sum(1 for r in self.ranks if r < k) / k

    def recall_at(self, k: int) -> float:
        if not self.total_relevant:
            return 0.0
        return sum(1 for r in self.ranks if r < k) / self.total_relevant

    @property
    def first_rank(self) -> int | None:
        return min(self.ranks) + 1 if self.ranks else None


def evaluate(chunks, store, embedder, queries=QUERIES, depth=10) -> list[QueryResult]:
    results = []
    for q in queries:
        total = sum(1 for c in chunks if q.is_relevant(c))
        hits = store.search(embedder.embed_query(q.question), k=depth)
        ranks = [i for i, h in enumerate(hits) if q.is_relevant(h.chunk)]
        results.append(
            QueryResult(
                query=q,
                total_relevant=total,
                ranks=ranks,
                top_source=hits[0].chunk.source if hits else "-",
                top_score=hits[0].score if hits else 0.0,
            )
        )
    return results


def aggregate(results: list[QueryResult]) -> dict[str, float]:
    n = len(results) or 1
    out = {}
    for k in KS:
        out[f"P@{k}"] = sum(r.precision_at(k) for r in results) / n
        out[f"R@{k}"] = sum(r.recall_at(k) for r in results) / n
        out[f"hit@{k}"] = sum(1 for r in results if r.hit_at(k)) / n
    ranked = [r.first_rank for r in results if r.first_rank]
    out["MRR"] = sum(1 / r for r in ranked) / n
    return out


def build(size: int, overlap: int, corpus="corpus", embedder=None,
          doc_prefix=DOC_PREFIX, query_prefix=QUERY_PREFIX):
    chunks = load_corpus(corpus, size=size, overlap=overlap)
    embedder = embedder or Embedder()
    store = NumpyStore()
    store.add(chunks, embedder.embed([c.with_context() for c in chunks],
                                     prefix=doc_prefix))
    return chunks, store, embedder


# printing


def print_table(results: list[QueryResult]) -> None:
    print(f"{'metric':<9}" + "".join(f"{f'k={k}':>9}" for k in KS))
    print("-" * (9 + 9 * len(KS)))
    agg = aggregate(results)
    for name in ("P@", "R@", "hit@"):
        label = {"P@": "precision", "R@": "recall", "hit@": "hit rate"}[name]
        print(f"{label:<9}" + "".join(f"{agg[f'{name}{k}']:>9.2f}" for k in KS))
    print(f"\nMRR: {agg['MRR']:.3f}   ({len(results)} queries)")


def print_per_query(results: list[QueryResult]) -> None:
    print(f"\n{'rank':>5}  {'rel':>4}  {'lex':<5} question")
    print("-" * 78)
    for r in sorted(results, key=lambda x: (x.first_rank or 99)):
        rank = str(r.first_rank) if r.first_rank else "MISS"
        q = r.query.question
        if len(q) > 52:
            q = q[:49] + "..."
        print(f"{rank:>5}  {r.total_relevant:>4}  {r.query.lexical_overlap:<5} {q}")


def print_split_by_overlap(results: list[QueryResult]) -> None:
    print("\nsplit by how much the question's wording matches the source:")
    print(f"{'group':<8}{'n':>4}{'hit@1':>9}{'hit@3':>9}{'hit@5':>9}{'MRR':>8}")
    print("-" * 47)
    for group in ("high", "low"):
        sub = [r for r in results if r.query.lexical_overlap == group]
        if not sub:
            continue
        a = aggregate(sub)
        print(f"{group:<8}{len(sub):>4}{a['hit@1']:>9.2f}{a['hit@3']:>9.2f}"
              f"{a['hit@5']:>9.2f}{a['MRR']:>8.3f}")
    print("\n'low' is questions phrased without the document's own vocabulary.")
    print("It is the more honest number -- I wrote both the docs and the")
    print("questions, and 'high' is flattered by that.")


def print_misses(results: list[QueryResult]) -> None:
    misses = [r for r in results if not r.hit_at(5)]
    if not misses:
        print("\nno misses at k=5.")
        return
    print(f"\n{len(misses)} queries with nothing relevant in the top 5:")
    for r in misses:
        print(f"  - {r.query.question}")
        print(f"      wanted {r.query.marker!r} in {'/'.join(r.query.sources)}; "
              f"top hit was {r.top_source} at {r.top_score:.2f}")
        if r.query.note:
            print(f"      note: {r.query.note}")


# ablations


def ablate_chunk_size(embedder: Embedder) -> None:
    print("\n" + "=" * 66)
    print("ABLATION 1 -- chunk size (overlap held at 1/6 of size)")
    print("=" * 66)
    print(f"{'size':>6}{'overlap':>9}{'chunks':>8}{'hit@1':>8}{'hit@3':>8}"
          f"{'hit@5':>8}{'R@3':>8}{'MRR':>8}")
    print("-" * 63)

    rows = []
    for size in (200, 300, 600, 1200, 2000):
        overlap = size // 6
        chunks, store, _ = build(size, overlap, embedder=embedder)
        res = evaluate(chunks, store, embedder)
        a = aggregate(res)
        rows.append((size, a))
        print(f"{size:>6}{overlap:>9}{len(chunks):>8}{a['hit@1']:>8.2f}"
              f"{a['hit@3']:>8.2f}{a['hit@5']:>8.2f}{a['R@3']:>8.2f}{a['MRR']:>8.3f}")

    best = max(rows, key=lambda r: (r[1]["hit@3"], r[1]["MRR"]))
    base = next(a for s, a in rows if s == 600)
    print(f"\nbest hit@3: size={best[0]} at {best[1]['hit@3']:.2f}")
    print(f"delta vs the configured 600: "
          f"{best[1]['hit@3'] - base['hit@3']:+.2f} hit@3, "
          f"{best[1]['MRR'] - base['MRR']:+.3f} MRR")
    print("\nWhy the shape: small chunks match sharply but often return the topic")
    print("sentence without the number underneath it; large chunks contain the")
    print("answer but their embedding is an average of several topics, so they")
    print("match everything mediocrely.")


def ablate_prefixes(embedder: Embedder) -> None:
    """Does nomic's asymmetric query/document prefix actually matter?

    I asserted in embed.py that skipping it is 'quietly worse'. That was a
    claim about behaviour I had read about, not measured, so this measures it.
    """
    print("\n" + "=" * 66)
    print("ABLATION 2 -- nomic-embed-text asymmetric prefixes")
    print("=" * 66)

    variants = [
        ("correct (doc/query)", DOC_PREFIX, QUERY_PREFIX),
        ("none at all", "", ""),
        ("same prefix both sides", DOC_PREFIX, DOC_PREFIX),
    ]

    print(f"{'variant':<24}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}")
    print("-" * 56)
    for label, dp, qp in variants:
        chunks = load_corpus("corpus", size=600, overlap=100)
        store = NumpyStore()
        store.add(chunks, embedder.embed([c.with_context() for c in chunks], prefix=dp))
        res = []
        for q in QUERIES:
            total = sum(1 for c in chunks if q.is_relevant(c))
            hits = store.search(embedder.embed([q.question], prefix=qp)[0], k=10)
            res.append(QueryResult(
                query=q, total_relevant=total,
                ranks=[i for i, h in enumerate(hits) if q.is_relevant(h.chunk)],
                top_source=hits[0].chunk.source if hits else "-",
                top_score=hits[0].score if hits else 0.0,
            ))
        a = aggregate(res)
        print(f"{label:<24}{a['hit@1']:>8.2f}{a['hit@3']:>8.2f}"
              f"{a['hit@5']:>8.2f}{a['MRR']:>8.3f}")




def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="measure retrieval on the Meridian corpus")
    ap.add_argument("--ablate", action="store_true", help="run the ablations too")
    ap.add_argument("--size", type=int, default=600)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--per-query", action="store_true", help="rank of every query")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    embedder = Embedder()
    chunks, store, _ = build(args.size, args.overlap, embedder=embedder)

    print(f"corpus: {len(chunks)} chunks (size={args.size} overlap={args.overlap})")
    print(f"store:  {store.stats()}")
    print(f"queries: {len(QUERIES)} hand-labelled\n")

    results = evaluate(chunks, store, embedder)
    print_table(results)
    print_split_by_overlap(results)
    print_misses(results)
    if args.per_query:
        print_per_query(results)

    if args.ablate:
        ablate_chunk_size(embedder)
        ablate_prefixes(embedder)

    print(f"\n[{time.perf_counter() - t0:.1f}s, {embedder.api_calls} embed calls, "
          f"{embedder.cache_hits} cache hits]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
