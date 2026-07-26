# Access Control

## Roles

Meridian has four roles. They are not hierarchical — `operator` is not a
superset of `analyst`.

- **reader** — can query gold tables through Lantern. Default for all
  employees.
- **analyst** — reader, plus silver tables and the ability to register
  non-production pipelines.
- **operator** — can trigger, pause and backfill pipelines, but *cannot read
  the data those pipelines produce*. This separation is deliberate.
- **owner** — full control of a specific dataset, including retention tier
  changes and granting analyst access to it.

The operator/reader split is the design decision people push back on most.
The reasoning: the population that needs to restart a stuck pipeline at 3am is
much larger than the population that should be able to read customer records,
and collapsing the two roles would have meant granting data access to the
entire on-call rotation.

## Break-glass

Emergency elevation is available through the break-glass path. It grants
`owner` on a named dataset for **60 minutes** and cannot be extended; a second
break-glass request for the same dataset within 24 hours requires director
approval.

Every break-glass activation posts to `#meridian-audit` immediately, pages the
dataset owner, and generates a review ticket that must be closed with a
written justification within 5 business days. Unclosed break-glass tickets
block the team's next quarterly access recertification.

## Audit logging

All Lantern queries are logged with the requesting identity, the datasets
touched, and the row count returned. Logs are retained for 400 days under the
`standard` tier.

Query *text* is logged only for queries against datasets tagged `regulatory`.
For everything else only the dataset list is recorded, because full query text
was found to contain customer identifiers pasted into WHERE clauses often
enough to be a liability of its own.
