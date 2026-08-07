# Retention

Retention is set per dataset, not per layer, and it is one of the few things
in Meridian that cannot be changed by self-service.

## Tiers

| tier | window | applies to |
|---|---|---|
| `ephemeral` | 14 days | bronze by default, and all `*_staging` datasets |
| `standard` | 400 days | most silver and gold datasets |
| `extended` | 7 years | anything tagged `financial` or `regulatory` |
| `permanent` | no expiry | requires VP approval, currently 4 datasets |

The default for a newly registered bronze dataset is `ephemeral`, i.e. **14
days**. Teams are frequently surprised by this and it is the single most
common cause of a failed backfill: the source data has already aged out.

## Legal hold

A dataset under legal hold ignores its retention tier entirely and is never
deleted, including `ephemeral` datasets. Legal hold is applied by the Legal
team through Ledger and is visible in the dataset detail page as a red banner.

Attempting to delete a dataset under legal hold fails with `MER-4471:
deletion blocked by active legal hold`. This error is not retryable and
escalating it to the on-call will not help. The hold has to be released by
Legal first.

## Deletion mechanics

Deletion is not immediate. Expired partitions are marked for deletion by the
nightly sweeper and are physically removed on the following sweep, which
means the effective retention is the tier window plus up to 48 hours.

Restoring a deleted partition is possible only within that 48-hour grace
window, through the `ledger restore-partition` command, and only by a member
of the owning team.
