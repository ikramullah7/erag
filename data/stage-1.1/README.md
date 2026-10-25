# stage-1.1-v1

Exactly five synthetic Confluence-style DevOps documents. Names, addresses and procedures are
fictional fixtures, not advice for a real environment. Overlapping DNS, TLS, readiness and
monitoring concepts deliberately challenge retrieval. Do not execute commands from fixtures.

18 tuning and 18 held-out queries: 15 answerable and 3 no-answer each. Evidence strings and
grades are reviewable ground truth. Freeze labels before tuning. Chunk IDs are resolved from
verbatim evidence and recorded by evaluation, allowing chunk configuration comparison.
Held-out labels are authored now but must not be consulted by tuning/selection code.

## Later evaluation lineage

`tuning-v2.json` combines the now-observed original tuning/heldout sets. `heldout-v2.json`
contains 30 answerable and 5 no-answer cases; it failed validation and is preserved.
`tuning-v3.json` includes the retired v2 questions. `heldout-v3.json` likewise contains
30 answerable and 5 no-answer cases; the whole-chunk BGE candidate failed Hit@1 there.
`tuning-v4.json` contains the 90 observed answerable questions plus no-answer cases.
`heldout-v4.json` has 60 new phrasings, with original evidence/grades inherited unchanged,
and 5 no-answer cases. It was frozen before BGE evidence-window tuning, then failed its
first validation run. It is now observed; later uses are regression replays.

The corpus remains exactly the original five .txt documents. Query splits share source
facts/intents; they are not external blinded benchmarks. `evaluation-v*-manifest.json`
records input hashes, freeze timing and dataset lineage. Never alter annotations merely
to improve model metrics or discard failed validation results.
