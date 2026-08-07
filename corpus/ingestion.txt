# Driftwood: Ingestion

Driftwood is the only supported way to get data into Meridian. Direct writes
to bronze storage are blocked at the IAM layer and will fail with
`MER-1003: unauthorized direct write`.

## Connectors

There are three connector classes:

- **Pull connectors** poll an external source on a schedule. Most connectors
  are of this type. The minimum poll interval is 5 minutes; anything shorter
  is rejected at registration.
- **Push connectors** accept records over the ingest HTTP endpoint. Rate
  limited to 20,000 records per second per connector, per region.
- **File drop connectors** watch an object-store prefix. These are the
  cheapest and the most common source of duplicate records.

## Batch windows

Driftwood batches records before landing them. The batch closes when either
condition is met, whichever comes first:

- 50,000 records accumulated, or
- 90 seconds elapsed since the batch opened.

Batches that close on the time condition rather than the size condition are
tagged `partial=true` in Ledger. A connector where more than 70% of batches
are partial is considered misconfigured and will raise a warning in the weekly
ingest report.

## The deduplication rule

This is the part people get wrong most often.

Driftwood deduplicates on a composite key of `(source_id, record_id,
event_time_bucket)` where the bucket is a **15 minute** floor of the event
timestamp. Two records with the same source and record id but event times
that fall in different 15-minute buckets are treated as **distinct records**,
not duplicates.

This was a deliberate choice after the Halberd incident, where a
strictly-by-id dedup silently dropped legitimate correction events. The
trade-off is that a source which retries across a bucket boundary will
produce a duplicate that Driftwood does not catch. Kettle is expected to
handle that case at the silver layer.

## Backfills

Backfills run through the same path as live ingestion but with the
`backfill=true` flag, which does three things: raises the batch size limit to
500,000 records, disables the partial-batch warning, and routes the run to the
dedicated `driftwood-backfill` compute pool so it cannot starve live traffic.

A backfill covering more than 30 days of history requires sign-off from the
Ingest on-call, because it will typically exceed the daily compute budget for
the team that requested it.
