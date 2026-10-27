# Current Stage 1.1 acceptance targets

The user's later instruction supersedes the earlier weaker chunk gates. Enforce all five:
Hit@1 >= .95, Hit@3 >= .98, Recall@5 >= .95, MRR >= .95, nDCG@10 >= .95.
Targets are runtime configuration in `config/quality.yaml`. Missing metrics fail closed.
Hit@k is per-query presence of any positive chunk in top k; Recall@k measures the fraction
of all labelled positive chunks retrieved. No-answer cases remain excluded and reported
separately. Document/latency/integrity requirements remain in force.

Previously observed held-out questions are now development data (`tuning-v2.json`), not
independent validation. Original labels and archives remain byte-for-byte unchanged.
New validation (`heldout-v2.json`) has 30 answerable and 5 no-answer questions; source facts
and relevance grades were frozen before the additional development sweep. The fresh set
is opened only after selection is persisted. Do not tune on its results or claim the old
benchmark is independent evidence for a newly selected strategy.

Historical methodology follows; its original thresholds and completion are superseded.

---

# Evaluation and acceptance

Approved by the user: maximum three tuning iterations, then freeze configuration and run
held-out evaluation once. Human-reviewable cases contain query, category, document relevance
grades and verbatim expected evidence grounded in the five source documents. Include exact
identifiers, semantic paraphrases, troubleshooting, ambiguity, multiple sources and no-answer cases.
Held-out cases cannot influence tuning. Report results separately for both splits.

Selected strategy held-out gates: document Recall@3 >= .90, MRR >= .85, nDCG@5 >= .85;
chunk Recall@5 >= .85 and nDCG@10 >= .80. All tests pass, zero index/chunk integrity failures.
Agreed warm CPU p95 latency: <= 2000 ms without reranking and <= 5000 ms with reranking.
Measure indexing and model-load duration separately. Hardware must accompany performance data.
These gates were approved and cannot be weakened silently.

Measure document Recall@3/5, MRR, nDCG@5 and chunk Recall@3/5/10, MRR, nDCG@5/10.
Use graded gain 2^grade - 1 and logarithmic rank discount. Document ranking deduplicates first
occurrence; recall denominator includes all relevant items. Report per-query failures and
category means; no-answer cases are excluded from answerable averages and reported explicitly.
At five documents document Recall@5 has little discriminative value. No abstention claim.

Compare BM25/vector/hybrid with and without reranking across three chunk configurations.
Use only retrieved candidates in reranking. Change one major parameter per tuning iteration.
Preserve corpus/config/label hashes, dependency lock, model commits, raw rankings, component
scores, final rankings and elapsed times. Validate every evidence label against source text.
Synthetic labels need human review; these results cannot establish production readiness.


## Executed protocol

The preregistered sweep selected among all six strategies using passing gates first, then chunk
nDCG@10, document nDCG@5, chunk Recall@5, lower p95 latency and smaller chunk size.
The best configuration was persisted before opening held-out labels. Exactly three tuning runs
and one held-out run executed. Overlap-only tests check integrity and preservation, without
retrieval tuning. Held-out labels must not guide future improvements without a fresh split.

Per-query misses in the report use a strict coverage check (document Recall@3 or chunk Recall@5
below 1), distinct from the aggregate quality thresholds. No missing relevant chunk is hidden by
a passing aggregate. Sources, labels and ranking settings were unchanged after evaluation.
Detailed tests and ranking replay validate subsequent diagnostics/compatibility maintenance.


## Five-document answer extension

Answer checks are separate from the historical retrieval metrics. The user accepted ranking
and authorized generation on the existing five sources. `answer-evaluation-v1.json` fixes 15
answerable questions and five unsupported/partially supported/adversarial questions, expected
facts and source-verified evidence before model comparison. These are development checks,
not a new held-out set. Prompt refinement uses the same questions; later runs are regressions,
not independent generalization evidence. No source facts or expected answers are changed.

Known-fact patterns check each requested fact; citation checks independently verify original
source offsets and exact quotations. Unsupported questions must abstain, including a question
with one supported and one unsupported material part. Numeric/quote validation can reject
outputs without proving semantic entailment. Manually inspect claims against their quotes,
particularly unsupported questions, rather than treating quote provenance as factual accuracy.

The CLI's engineering smoke checks require >=90% known-fact cases, 100% correct abstentions,
100% citation provenance among answered cases and zero invalid/provider errors. These are
implementation checks, not newly user-agreed retrieval targets or production acceptance.
Each run preserves individual failures, configurations, digests, prompt version, source/dataset/
chunk hashes, runtime usage and timing. Rejected generations are retained only in evaluation
diagnostics and never rendered as factual CLI answers. Earlier failed runs are not overwritten.

The model comparison supplies identical retrieved evidence to both models. Report generation
median/p95 separately; cached retrieval timings are not fresh end-to-end measurements. Local
hardware, warm-up/model loading and runtime versions affect results. A 20-case development
check is insufficient to claim production readiness or approve the next document stage.


Real-five repair regression v10: original 20, second 20, two shared unsupported controls and one
separately versioned user clarification total43 unique cases. Freeze clarified restart-policy /
ReplicaSet requirements before generation; preserve original q02. New evaluations of observed
questions are development checks. Recorded canonical initial rankings, fresh bounded fact search
and actual local 14B generations are distinct from the subsequent model-free section replay.

Completion must preserve every original cited assertion in a same-document/version superset.
Its model_source_ids bind expanded_context to pre-completion sources; actual saved catalog
binds model_visible; final selected_quotes may include post-generation sections. Never credit
post-generation context as model input. Exact witness recognition is not semantic equivalence.

Primary-agent answer annotations remain provisional. Sentence/row decomposition includes
extra factual background; long tables group adjacent rows within the existing judgment schema.
Missing required facts are penalized despite truthful extra content. Extractive copying makes
faithfulness/citation scores close to source fidelity and does not measure useful synthesis,
concision or question relevance. Group scores reuse the same two negative controls; they do not
constitute four independent negatives. No new acceptance thresholds or held-out claims apply.


Brief answer checks (Codex active profile) additionally enforce a word budget and a same-provider
support/coverage/focus verdict. Required parts are judged against the question, not every background
fact in the selected excerpts. Reject omitted conditions or repeated/unrelated content; retain
one failed semantic attempt if revised. Exact quote provenance remains a separate deterministic
check. Unit coverage exercises missing qualifications, unsupported facts, malformed verdicts,
word overflow, model changes, review failures, bounded revision and evidence inspection. Two
live smoke cases do not replace a fresh question/reference benchmark or establish new aggregate
accuracy. Historical extractive local14B scores remain historical.
