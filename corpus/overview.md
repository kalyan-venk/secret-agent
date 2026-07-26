# Meridian — Overview

Meridian is the internal data platform operated by the Platform Reliability
group. It replaced the previous system, Cascade, which was decommissioned in
March 2024 after the Halberd incident.

## The four subsystems

Meridian is not one service. It is four, and they fail independently:

1. **Driftwood** — ingestion. Pulls from external connectors and lands raw
   records in the bronze layer. Owned by the Ingest team, on-call rotation
   `pr-ingest`.
2. **Kettle** — transformation. Runs the declared pipelines that promote
   bronze to silver and silver to gold. Owned by the Pipelines team.
3. **Lantern** — the query layer. Everything user-facing reads through
   Lantern; nothing queries storage directly. Owned by the Serving team.
4. **Ledger** — the metadata and lineage catalogue. Knows which dataset came
   from which pipeline run. Owned jointly by Pipelines and Serving, which has
   been a source of friction since the split in 2024.

## Layers

Meridian uses a three-layer model. Bronze is raw and immutable. Silver is
cleaned and deduplicated. Gold is aggregated and is the only layer external
teams are permitted to query.

The rule that surprises people: **a gold table may never read from bronze
directly.** Every gold table must have a silver ancestor. This is enforced by
Ledger at pipeline registration time and violations are rejected with error
code `MER-2200`.

## Ownership and the RFC process

Any change that alters a gold table's schema requires an RFC filed under the
`MER` prefix and approved by two reviewers, at least one from Serving. The
current RFC backlog target is 5 business days to first review.

Schema changes to silver tables do not need an RFC but do need a heads-up in
`#meridian-changes` at least 24 hours before deploy.

## Scale, as of the last capacity review

Meridian processes roughly 41 billion records per day across 312 registered
pipelines. The largest single pipeline, `driftwood.clickstream.raw`, accounts
for about 38% of that volume on its own.
