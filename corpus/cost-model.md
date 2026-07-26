# Cost Model

Meridian charges back to teams monthly. Two components: storage and compute.

## Storage

Billed on average bytes stored across the month, at the layer's rate:

- bronze: 1.0x base rate
- silver: 1.4x base rate (the premium covers the dedup index)
- gold: 2.2x base rate (covers replication to all three regions)

Gold is expensive on purpose. A gold table replicated to three regions that
nobody queries is the most common source of surprise cost, and the quarterly
review specifically looks for gold tables with fewer than 10 queries a month.

## Compute

Billed in pipeline-seconds, weighted by the compute pool:

- `driftwood-live` — 1.0x
- `driftwood-backfill` — 0.4x (cheaper, but preemptible and not latency
  guaranteed)
- `kettle-standard` — 1.0x
- `kettle-highmem` — 3.1x

The `kettle-highmem` multiplier is the reason pipeline authors are asked to
justify highmem in the RFC. A pipeline that could run on standard but was
written for highmem out of caution is a 3x cost multiplier for the life of
the pipeline.

## Budgets and enforcement

Each team gets a monthly compute budget set at the start of the quarter.
Exceeding it does **not** stop pipelines — that was considered and rejected,
because the failure mode of a halted production pipeline is worse than the
cost overrun. Instead, exceeding the budget by more than 20% moves all of the
team's non-tier1 pipelines to preemptible pools for the remainder of the
month, which slows them down without stopping them.

Tier1 pipelines are exempt from that. The exemption list is reviewed
quarterly and currently has 23 pipelines on it.
