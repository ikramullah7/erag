# Experiments

Three approved tuning iterations completed. No fourth tuning iteration.

| Run | Words/overlap | Chunks | Index ms | Strategy | Doc nDCG@5 | Chunk R@5 | Chunk nDCG@10 | Missed queries |
|---|---:|---:|---:|---|---:|---:|---:|---|
| tuning-1 | 100/20 | 30 | 261.1 | bm25 | 0.989 | 0.956 | 0.946 | tuning-07, tuning-10 |
| tuning-1 | 100/20 | 30 | 261.1 | bm25+rerank | 0.989 | 1.000 | 0.930 | none |
| tuning-1 | 100/20 | 30 | 261.1 | vector | 0.912 | 0.922 | 0.817 | tuning-07, tuning-15 |
| tuning-1 | 100/20 | 30 | 261.1 | vector+rerank | 0.989 | 1.000 | 0.930 | none |
| tuning-1 | 100/20 | 30 | 261.1 | hybrid | 0.945 | 0.978 | 0.893 | tuning-07 |
| tuning-1 | 100/20 | 30 | 261.1 | hybrid+rerank | 0.989 | 1.000 | 0.930 | none |
| tuning-2 | 140/20 | 29 | 247.4 | bm25 | 0.978 | 0.978 | 0.959 | tuning-07 |
| tuning-2 | 140/20 | 29 | 247.4 | bm25+rerank | 0.989 | 1.000 | 0.932 | none |
| tuning-2 | 140/20 | 29 | 247.4 | vector | 0.936 | 0.922 | 0.843 | tuning-07, tuning-15 |
| tuning-2 | 140/20 | 29 | 247.4 | vector+rerank | 0.989 | 1.000 | 0.932 | none |
| tuning-2 | 140/20 | 29 | 247.4 | hybrid | 0.945 | 0.978 | 0.910 | tuning-07 |
| tuning-2 | 140/20 | 29 | 247.4 | hybrid+rerank | 0.989 | 1.000 | 0.932 | none |
| tuning-3 | 180/20 | 15 | 134.6 | bm25 | 0.986 | 1.000 | 0.953 | none |
| tuning-3 | 180/20 | 15 | 134.6 | bm25+rerank | 0.978 | 0.967 | 0.950 | tuning-08 |
| tuning-3 | 180/20 | 15 | 134.6 | vector | 0.962 | 0.911 | 0.812 | tuning-06, tuning-07, tuning-15 |
| tuning-3 | 180/20 | 15 | 134.6 | vector+rerank | 0.978 | 0.967 | 0.950 | tuning-08 |
| tuning-3 | 180/20 | 15 | 134.6 | hybrid | 0.986 | 0.967 | 0.903 | tuning-15 |
| tuning-3 | 180/20 | 15 | 134.6 | hybrid+rerank | 0.978 | 0.967 | 0.950 | tuning-08 |

140/20 BM25 won the predefined tuning selection rule (chunk nDCG@10 .959).
Increasing word budget to 180 reduced chunk count to 15 but did not improve the selected ranking metric.
Reranking often improved recall but did not uniformly improve graded ranking; tuning-08 regressed at 180 words.
The held-out evaluation was run once after freezing selection. No labels or ranking parameters changed afterward.

All runs: dataset stage-1.1-v1, exact models/revisions and all config values in reports,
candidate count 20, RRF k=60, overlap 20. Overlap-only comparisons were integrity audits.
See [stage report](stage-reports/stage-1.1.md) and [raw/compact results](results/README.md).


## Answer extension: fixed local-model comparison

`artifacts/answer-evaluation-v1`: same 20 questions and frozen full-context BGE retrieval for
both installed models. Prompt `grounded-claims-v1`. Qwen3 14B: 14/15 known-fact cases,
5/5 correct abstentions, one rejected answer, generation median 10.58 s. Qwen3 8B: 15/15
known-fact cases, 4/5 abstentions, no rejected answers, median 6.08 s. Both smoke gates failed.
Every displayed citation had exact provenance, demonstrating that provenance alone does not
prove semantic support: 8B answered a two-part question's known half and cited networking
text for a statement about an absent password. 14B added a cluster identifier absent from
its particular citation quote; the validator correctly rejected it. Its isolated replay and
raw model payload are in `artifacts/answer-14b-failure-replay.json`.

`grounded-claims-v2` strengthens generic all-parts coverage, minimal answers and citation-local
identifier support. Dataset, expected facts, documents, retrieval, model digests and generation
parameters remain fixed. The unchanged 20-case comparison is rerun under
`artifacts/answer-evaluation-v2`. Earlier source is archived with each comparison for replay.
This is prompt refinement on observed development questions, not a new validation sample.


Revised-prompt results: both models pass 15/15 known-fact cases, 5/5 correct abstentions,
100% quote provenance and zero invalid/provider errors. Generation medians: 14B 9.67 s,
8B 6.11 s. Retain 14B as the initial default; 8B is a faster selectable alternative.
These figures measure expected-fact checks rather than a formal semantic-accuracy score.
All returned claims/quotes were inspected by the primary agent; review limitations are recorded
in `docs/results/answers-phase-1.1/manual-review.json`. Compressed raw cases, frozen inputs,
source snapshots, both failed and passing comparisons are archived in the same directory.


## Stage 1.2 frozen expansion

Ten documents/52 chunks; retained original five and fixed accepted ranking profile. 36 answer
cases: fifteen legacy regressions, fifteen new supported questions, six unsupported cases.
Sources/labels/reference facts were reviewed before first retrieval. All six ranking strategies
ran on frozen tuning/heldout splits, with no parameter changes after observing either split.
Selected existing hybrid+BGE heldout: Hit@1/Hit@3 1.0, Recall@5 .9791667, MRR 1.0,
nDCG@10 .9860. All configured ranking targets pass. Full strategy summaries and failures are
archived; vector+BGE descriptively had higher heldout recall and was not selected from heldout.

Initial local 14B generation: 28/30 supported cases, 6/6 abstentions, two blocked quote/identifier
validation failures. Repaired output validation and exact-source fragment alignment are enabled
only for stage 1.2. Failed repair attempts and alignment bug regression are preserved. Final
replay: 30/30 expected-fact checks and 6/6 abstentions; measured first attempts reused, one fresh
ownership repair requested. A fresh real CLI answer on the ten-source index passed. Quote
alignment records exact source offsets and original model strings, preserving semantic review.

Agent-reviewed final baseline: faithfulness 72/73, correctness mean F1 .972222, citation support
68/70, abstention 36/36. Three findings and conservative reference-scope decisions are recorded.
No source/reference fact was edited after outputs to raise scores. Evidence and input/code
snapshots: `docs/results/stage-1.2`. These are synthetic development and regression observations,
not new independent human validation or production readiness.


## Stage 1.3 fixed twenty-document expansion

Twenty sources/82 chunks; ten retained files and 52 chunk IDs unchanged. Fixed ranking profile;
14B generation context 12,288 set before evaluation. Frozen questions: ten prior regressions,
twenty new supported questions, six unsupported. Tuning fifteen supported; heldout fifteen
supported plus six unsupported. All six retrieval variants executed. Selected existing hybrid+
BGE: heldout Hit@1/Hit@3/Recall@5/MRR/nDCG@10 all 1.0; tuning nDCG@10 .9865 with other selected
metrics 1.0. No heldout-based strategy selection or ranking tuning.

Fresh 14B answers: 30/30 expected-fact checks, 6/6 correct abstentions, one successful bounded
retry (37 API calls), zero final invalid/provider errors. No recorded-answer replay. A separate
fresh CLI answer also passed. Generation-only median 10.66 s; ranking timing reuse is explicit
and not a fresh full-pipeline latency measurement.

Agent-reviewed semantics: faithfulness 71/74, correctness mean F1 .962525, citation accuracy
67/70, abstention 36/36. Per-case and grouped scores retain old certainty issues, unsupported
Loki backend assignment, a conservative high-severity clock-anchor interpretation and a
compound-reference scope caveat. No source/expected fact was rewritten after seeing answers.
These are agent-reviewed synthetic development observations, not independent human gold scores.
All raw outputs, source snapshots, annotations, frozen hashes and tests: `docs/results/stage-1.3`.
