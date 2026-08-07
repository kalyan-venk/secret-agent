"""Hand-labelled query -> passage pairs.

## How gold is labelled, and why it isn't a chunk id

The obvious labelling is "query 4 should retrieve chunk 12". That breaks the
moment you run the ablation, because changing chunk size renumbers every
chunk in the corpus -- chunk 12 at size 600 is different text from chunk 12
at size 300. You would be comparing against a moving target and the
comparison would be meaningless.

So gold is (source document, marker string) instead. A retrieved chunk counts
as relevant if it came from the right document AND contains the marker. That
is stable across any chunking parameters, which is what makes the ablation
possible at all.

The marker is always the specific fact the question asks for -- "14 days",
"MER-4471", "3.1x". Not a topic word. A chunk that discusses retention
without stating the window has not answered the question and should not score.

## lexical_overlap

I wrote both the documents and the questions, which is a real weakness: it is
easy to unconsciously phrase a question in the source's own words, and
embedding search then looks better than it would on questions from someone
who had not read the docs.

So each query is tagged `high` or `low` for how much vocabulary it shares
with the passage, and eval.py reports the two groups separately. The `low`
questions are the honest ones. If there is a big gap between the groups,
the headline number is flattered by my phrasing and the `low` number is
closer to the truth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuery:
    question: str
    sources: tuple[str, ...]   # documents that can legitimately answer it
    marker: str                # the specific fact, case-insensitive
    lexical_overlap: str = "high"
    note: str = ""

    def is_relevant(self, chunk) -> bool:
        return (
            chunk.source in self.sources
            and self.marker.lower() in chunk.text.lower()
        )


QUERIES: list[EvalQuery] = [
    EvalQuery(
        "What is the default retention window for a newly registered bronze dataset?",
        ("retention.txt",), "14 days", "high",
    ),
    EvalQuery(
        "How long does break-glass access last?",
        ("access-control.txt",), "60 minutes", "high",
    ),
    EvalQuery(
        "What error code is returned when deleting a dataset under legal hold?",
        ("retention.txt",), "MER-4471", "high",
    ),
    EvalQuery(
        "When does Driftwood close a batch?",
        ("ingestion.txt", "glossary.txt"), "50,000 records", "high",
    ),
    EvalQuery(
        "What is the event time bucket size used for deduplication?",
        ("ingestion.txt", "glossary.txt"), "15 minute", "high",
    ),
    EvalQuery(
        "How much did page volume drop after the ten minute requirement was added?",
        ("oncall.txt",), "82%", "high",
    ),
    EvalQuery(
        "What is the compute cost multiplier for kettle-highmem?",
        ("cost-model.txt",), "3.1x", "high",
    ),
    EvalQuery(
        "What is the p99 query latency threshold that triggers a page?",
        ("oncall.txt",), "2.5 seconds", "high",
    ),
    EvalQuery(
        "What is the rate limit for push connectors?",
        ("ingestion.txt",), "20,000 records per second", "high",
    ),
    EvalQuery(
        "What percentage of daily volume does the clickstream pipeline account for?",
        ("overview.txt",), "38%", "high",
    ),
    EvalQuery(
        "What error code fires when a gold table tries to read from bronze?",
        ("overview.txt",), "MER-2200", "high",
    ),
    EvalQuery(
        "How long is the grace window for restoring a deleted partition?",
        ("retention.txt",), "48", "high",
    ),
    EvalQuery(
        "What replaced 'ledger describe --legacy' and what changed about the output?",
        ("deprecations.txt",), "--json", "high",
    ),

    # --- low lexical overlap: phrased the way someone who hadn't read the
    # docs would phrase it
    EvalQuery(
        "If I try to remove data that Legal has frozen, what comes back?",
        ("retention.txt",), "MER-4471", "low",
        note="says 'frozen' and 'remove', docs say 'legal hold' and 'deletion'",
    ),
    EvalQuery(
        "How quickly do emergency elevated permissions go away on their own?",
        ("access-control.txt",), "60 minutes", "low",
        note="never says break-glass",
    ),
    EvalQuery(
        "Which people can bounce a stuck job at 3am without seeing customer records?",
        # WAS the bare substring "operator", which appears 3x in the document
        # and tagged chunks that merely mention the role as relevant. A marker
        # has to identify the passage that ANSWERS the question, not the topic.
        ("access-control.txt",), "cannot read", "low",
        note="describes the role by its purpose, not its name",
    ),
    EvalQuery(
        "My overnight job keeps stalling on one bad row. What does the system do with it?",
        ("oncall.txt", "glossary.txt"), "quarantin", "low",
        note="'bad row' vs 'poison record'",
    ),
    EvalQuery(
        "Someone told me bouncing the transform workers clears a backlog. Is that right?",
        ("oncall.txt",), "restart Kettle workers", "low",
        note="'transform workers' vs 'Kettle'; the doc says explicitly not to",
    ),
    EvalQuery(
        "What happens to a team that blows through its monthly spend?",
        ("cost-model.txt",), "preemptible pools for the remainder", "low",
        note="'blows through its spend' vs 'exceeding the budget'",
    ),
    EvalQuery(
        "Why did they stop using the old platform?",
        # WAS ("deprecations.txt", "glossary.txt") -- but Halberd appears in
        # overview.txt, glossary.txt and ingestion.txt, and NOT in deprecations.txt
        # at all. The gold label named a document that could not satisfy it.
        # Caught in review 2026-07-25; see test_gold_marker_is_present_in_every_named_source,
        # which now makes this class of error impossible to reintroduce.
        ("overview.txt", "glossary.txt"), "Halberd", "low",
        note="requires connecting 'old platform' -> Cascade -> Halberd",
    ),
]


def by_overlap(group: str) -> list[EvalQuery]:
    return [q for q in QUERIES if q.lexical_overlap == group]
