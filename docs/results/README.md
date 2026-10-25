# Recorded Stage 1.1 measurements

`evaluation-summary.json` contains configuration, versions, metrics, category summaries, query text,
exact labelled chunk IDs/grades and final rankings for all four measured runs. Full text and
component rankings/scores are retained in `tuning-*.json.gz` and `heldout.json.gz`.

Read a raw report using `python3 -m gzip -dc docs/results/heldout.json.gz`.
`archive-manifest.json` records compressed and original SHA-256 hashes. All archives contain
actual executions, not synthetic metrics. `tests.log` and `tests.xml` record executed tests and skips.

The held-out set must not be reused for future tuning. The regression replay uses only tuning queries.
The quality-gate archive measures retrieval thresholds; the overall stage gate also requires
the operational checks described in the stage report.
