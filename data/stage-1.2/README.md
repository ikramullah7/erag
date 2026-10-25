# Ten-document stage 1.2

The original five `.txt` files are byte-for-byte retained from stage 1.1. The five added files
cover PostgreSQL recovery, Redis availability, workload identity, releases and incidents.
`stage-manifest.json` records original-source hashes and evaluation input hashes.

`answer-evaluation-v1.json` contains 36 fixed questions: 15 legacy regressions, 15 new questions
and six unsupported questions. `answer-semantic-references-v1.json` adds explicit required facts
for semantic review. The original stage-1.1 files/labels are unchanged. PostgreSQL retention and
Redis failover, previously unknown, are answerable in this corpus; actual passwords and current
backup/replica/release state remain unknown.

`tuning.json` and `heldout.json` contain disjoint fixed retrieval question sets. Both were frozen
before retrieval; the existing stage-1.1 profile was retained without tuning on either split.
Legacy regression questions are not fresh evidence. New questions/labels are agent-authored,
source-verified development fixtures, not independent human validation. Scope/version checks
exercise archived staging versus current production policies. Cross-document checks exercise
identity, DNS and recovery scopes.

Ranking evidence includes complete alternative answers introduced by new sources. Grade 3
marks complete answer-bearing evidence; grade 2 marks required partial evidence for multi-part
questions. Quotes must be exact source spans within a canonical chunk. Incidental topic mentions
are not relevant answers. The source-first label review happened before any ranking output was
inspected; it is recorded in the manifest. All six retrieval variants are evaluated separately;
no-answer cases are excluded from relevance means and scored for abstention separately.

Run with explicit profiles; the old defaults remain available for stage-1.1 replay:

```sh
HF_HUB_OFFLINE=1 uv run erag --config config/stage-1.2.yaml index
HF_HUB_OFFLINE=1 uv run erag answer 'What is production PostgreSQL backup retention?' --answer-config config/answer-stage-1.2.yaml
```

No next document stage is authorized. Synthetic policy documents do not establish live service
health, current secrets, actual deployed versions or successful recovery outcomes.
