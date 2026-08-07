# Deprecations

## Removed

**Cascade**: the predecessor platform. Decommissioned March 2024. Any
document referring to Cascade paths (`cascade://`) is out of date. There is
no migration tooling left; the migration window closed.

**The v1 ingest endpoint**: removed January 2025. Push connectors must use
`/v2/ingest`. Requests to v1 return `MER-1099: endpoint removed` rather than
a 404, deliberately, so the error is greppable.

## Deprecated, still working

**`ledger describe --legacy`**: the old flat output format. Still works,
will be removed at the end of Q3. The replacement is `ledger describe --json`,
which is not a drop-in: the legacy format printed retention as a human string
("14 days") and the JSON format returns retention as an integer number of
days, so anything parsing the old output will silently get a different type.

**Region `eu-west-legacy`**: being drained. New datasets cannot be created
there. Existing ones are being moved on a schedule; the last move is planned
for the end of the calendar year. Datasets in this region are billed at the
gold rate regardless of their actual layer, which is intentional and is meant
to encourage migration.

**Pipeline definitions in YAML**: the Python DSL is the supported path now.
YAML pipelines still run but cannot use any feature added after the DSL
switch, which includes conditional stages, fan-out, and the retry policy
block. There is no automatic converter and there are no plans to write one;
the 40-odd remaining YAML pipelines are expected to be rewritten by their
owners.

## Not deprecated, despite rumours

`driftwood.quarantine` is not going away. There was a proposal to remove it in
favour of a dead-letter topic, it was rejected, and the confusion persists.
