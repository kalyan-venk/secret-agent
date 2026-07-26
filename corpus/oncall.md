# On-call and Runbook

## Rotations

Four rotations, matching the four subsystems: `pr-ingest`, `pr-pipelines`,
`pr-serving`, `pr-catalog`. Shifts are one week, handover Wednesday at 10:00
local to the outgoing engineer.

## Paging thresholds

A page fires when any of these hold for 10 consecutive minutes:

- Lantern p99 query latency above **2.5 seconds**
- Driftwood ingest lag above **15 minutes** on any connector tagged `tier1`
- Kettle pipeline failure rate above **5%** across a rolling 1 hour window
- Ledger write availability below **99.5%**

The 10-minute requirement was added after the on-call load review in 2024
found that 61% of pages were self-resolving within 4 minutes. Page volume
dropped by roughly 82% after the change with no measurable increase in
incident duration.

## First actions for ingest lag

1. Check whether the lag is on one connector or many. One connector is
   usually a source-side problem and is not yours to fix.
2. If many, check the `driftwood-live` compute pool for saturation. A
   backfill that was not routed to `driftwood-backfill` is the usual cause.
3. If the pool is healthy, check for a poison record — Driftwood will retry a
   malformed record 3 times before quarantining it, and a burst of them
   stalls the batch.
4. Quarantined records land in `driftwood.quarantine` and are kept for 14
   days. They are not automatically reprocessed; someone has to decide.

## Escalation

Escalate to the subsystem lead after 30 minutes without a diagnosis, or
immediately if the incident involves data loss or a suspected access-control
failure. Do not wait on an access-control incident. The escalation path for
those bypasses the subsystem lead and goes directly to the security on-call.

## What not to do

Do not restart Kettle workers to clear a backlog. It looks like it helps
because the queue depth drops, but in-flight batches are lost and reprocessed,
which doubles the load about 90 seconds later. This has caused two incidents.
