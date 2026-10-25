# Current result: Phase 1.3 implementation and evaluation complete

Twenty approved documents and 82 chunks. All ten prior source bytes and 52 chunk IDs retained.
Profiles: `config/stage-1.3.yaml`, `config/answer-stage-1.3.yaml`. Explicit guards reject stage
1.4/unauthorized source sets. Fixed ranking/chunking parameters; only the new answer context
budget increased to 12,288 before evaluation. Local 14B only, existing output/repair/alignment
policy retained. No new dependencies, model downloads or service integrations.

Thirty supported questions (ten regressions, twenty new) and six unsupported questions were
frozen before retrieval. Tuning has fifteen supported; heldout fifteen supported plus six
unsupported. All six real retrieval variants executed. Fixed hybrid+BGE heldout metrics:
Hit@1/Hit@3/Recall@5 100%, MRR/nDCG@10 1.0. All five configured ranking targets pass. No
ranking tuning or strategy selection on heldout results. Labels/sources are agent-authored
synthetic development fixtures with possible fact overlap, not independent human validation.

All 36 answer cases freshly generated, including one bounded validation retry (37 local API
calls). Fact patterns 30/30, unsupported abstentions 6/6, no final invalid/provider errors.
Generation timing is fresh; cached retrieval diagnostics reuse the actual ranking run. Do not
interpret evaluation elapsed_ms as fresh end-to-end latency. A separate fresh twenty-document
CLI generation for secret mount path/polling passed. No credential value was requested/returned.

Direct agent-reviewed baseline: faithfulness 71/74 (95.95%), correctness macro F1 96.25%,
semantic citation support 67/70 (95.71%), abstention 36/36 (100%). Findings: regression-03
requested/actual certificate validity; fresh-07 unsupported Loki audit-storage assignment;
fresh-19 conservative high-severity clock-anchor interpretation. Fresh-10 correctly answers the
asked no-workload-stop question but misses an unasked spending-cap detail in a compound
reference; strict reference scope is explicitly recorded rather than crediting reference text
as answer text. Fresh-11 has a served-hostname phrasing advisory without a fact-field penalty.
Original answers, sources and expected facts remain unchanged. Scores are reviewed development
estimates; no new semantic threshold/production pass or independent human agreement claimed.

Verification: 172 passed, two optional checks skipped (Docker server parity and historical 1.1
strict gate). Real cached model tests executed. Stage 1.1/1.2/1.3 real stored indexes all validate.
Ruff check/format and whitespace checks passed. Every claim, citation and required fact has a
hash-bound annotation; replay is deterministic and requires no local model. Evidence:
`docs/results/stage-1.3/summary.json`, sibling raw/code/test archives and `data/stage-1.3`.

Requested stage complete. Next action: user review/discussion. Stage 1.4 with fifty documents
remains unapproved. No narrative stage report generated.

---

## Historical work log

# Current work: Phase 1.3 — twenty documents

User explicitly authorized the twenty-document stage. All ten prior source files and their
52 chunk IDs are retained unchanged. Ten added runbooks cover messaging, vault rotation,
regional recovery, storage access, capacity, ingress, secret delivery, GitOps, logging governance
and patch compliance. The twenty-document index has 82 chunks. No later stage is authorized.

Profiles: `config/stage-1.3.yaml`, `config/answer-stage-1.3.yaml`. Retrieval parameters and pinned
models are unchanged. Qwen3 14B only; generation context was fixed at 12,288 before first query
to accommodate more source evidence and bounded repair feedback. Previous profiles untouched.
No dependency, model download, external service or interface expansion.

Frozen checks: ten retained regression questions, twenty new supported questions and six
unsupported cases. Tuning has fifteen supported questions; heldout has fifteen supported plus
six unsupported. Required facts, exact source spans and source-first labels were frozen before
retrieval. Questions/labels are agent-authored synthetic development data, not independent
human validation. Same source facts can appear across questions; no ranking tuning or selection
was performed using heldout results.

Completed: twenty-source isolation, retention/stable-ID, deletion/restore, stale-source and
source-evidence tests. Real six-strategy ranking evaluation completed. Fixed hybrid+BGE heldout
Hit@1/Hit@3/Recall@5 100%, MRR/nDCG@10 1.0; all configured ranking targets pass. Tuning nDCG@10
.9865, other selected metrics 1.0. In progress: fresh 14B generation, direct source-reviewed
four-metric baseline, final real-provider regressions and CLI smoke.

---

# Current result: Phase 1.2 implementation and evaluation complete

Ten approved documents, 52 canonical chunks. The original five source bytes and their chunk
IDs are unchanged. New explicit stage guards reject unapproved document sets/stages. Profiles:
`config/stage-1.2.yaml`, `config/answer-stage-1.2.yaml`. Qwen3 14B only for generation; cached
embedding/BGE models reused. No new dependency, model download or service introduced.

Frozen ranking evaluation: all six strategies, fixed configuration before first query; tuning
18 answerable cases and heldout 12 answerable plus six unsupported cases. Existing hybrid+BGE
heldout result: Hit@1 100%, Hit@3 100%, Recall@5 97.92%, MRR 1.0, nDCG@10 .9860. All five
existing configured thresholds pass. One multi-part question (new-28) retrieved three of four
alternative relevant chunks in the top five; sufficient evidence for a correct answer remained.
No heldout-based profile selection or parameter tuning was performed.

Answer checks: 30/30 supported cases and 6/6 correct abstentions, zero final invalid/provider
errors. All first-pass generations were actual local 14B calls. Two initial outputs were
rejected. One bounded repair is enabled for 1.2 and quote fragments may be aligned only to
unique exact source spans with initial-case/terminal-punctuation differences. No lexical or
numeric rewriting, no evidence/label changes. Phase 1.1 defaults remain strict with no retries.

The initial attempted repair lacked context space (repair-001). Compact failure-only feedback
repaired ownership but Redis quote fragments still failed (repair-002). A set-mutation bug in
initial alignment was caught by regressions and fixed; the failed replay is retained
(repair-003). Final replay repair-004 revalidates the measured first-pass outputs, aligns the
Redis fragments and makes one fresh ownership repair call. A fresh real CLI generation on
the ten-document index independently exercised the alignment path and passed. Replayed
elapsed_ms is not fresh end-to-end latency; measured generation timings and reuse are explicit.

Direct agent review: faithfulness 72/73 (98.63%), correctness mean F1 97.22%, citation support
68/70 (97.14%), abstention 36/36 (100%). Findings: legacy-04 requested/actual certificate-validity
qualification; new-18 a partial comparison citation; new-26 missing separately reviewed database
plan required by the compound reference fact (strict reference scope is explicitly reviewable).
No answer/source/expected fact was changed to increase semantic scores. No new semantic
acceptance threshold or independent human validation claim.

Verification: 167 passed, two optional checks skipped (Docker server parity and historical
1.1 strict quality regression). Real cached retrieval provider tests executed. Ruff check,
format and whitespace checks passed. All data, configurations, original failures, final
answers/reviews, frozen hashes and code snapshots: `docs/results/stage-1.2/summary.json` and
its sibling artifacts. New case labels and reviewed annotations: `data/stage-1.2`.

This requested stage is complete. Next action is user review/discussion; Phase 1.3 with twenty
documents remains unapproved. No narrative stage report was generated.

---

## Historical work log

# Current work: Phase 1.2 — ten documents

User explicitly authorized Phase 1.2. Exactly ten synthetic source documents, retaining the
original five byte-for-byte. Added PostgreSQL recovery, Redis availability, workload identity,
release management and incident coordination. No document stage beyond ten is authorized.

Implemented explicit stage guards and isolated profiles/index paths; retained five-source
chunk IDs, updates/deletion/restore and old-stage replay. Configuration:
`config/stage-1.2.yaml` and `config/answer-stage-1.2.yaml`. Generation remains the pinned local
Qwen3 14B, with the accepted unchanged retrieval/chunking/reranker profile. No dependencies,
models or cloud services added.

Frozen checks: 36 answer questions (15 prior regressions, 15 new questions, six unsupported),
with tuning/heldout retrieval splits frozen before any retrieval query. Full alternative-source
label review added new evidence for readiness/closure/retention before evaluation. PostgreSQL
retention and Redis failover are now answerable; original Phase 1.1 labels remain unchanged.
Fresh checks include scope/version differences, partial questions, missing live state and
cross-document information. New source-verified reference facts are versioned under stage-1.2.

In progress: real six-strategy retrieval evaluation and 14B generation; then direct source-
reviewed four-metric baseline and full regressions. Five new stage-isolation tests passed.
All results/failed cases will be retained. No numerical quality target was silently changed;
compare descriptive results against existing configured ranking targets. No narrative stage
report requested or generated.

---

# Current result: agent-reviewed five-document answer baseline

User authorized source-reviewed baseline scores and explicitly removed the unsuccessful-
iterations approval rule. This does not authorize new dependencies, document stages or paid
services. No local judge was run for the reviewed baseline; generation remains Qwen3 14B.

All 20 saved answers, 30 atomic assertions, 29 citation links and required facts were reviewed
against the five original source documents and retrieved excerpt context. Faithfulness: 29/30
(96.67%). Answer correctness: macro F1 96.67% (14 fully correct and one half-credit answer out
of 15 answerable cases). Semantic citation accuracy: 28/29 (96.55%). Abstention accuracy: 20/20
(100%), including five appropriate abstentions and no false abstentions.

One issue: answer-04 states actual 90-day validity while the source establishes requested
certificate duration and instructs checking the actual served certificate. The conservative
qualifier judgment and all supporting/coverage reasons are explicit in versioned annotations.
No answer, source document or reference fact was changed. Exact quote provenance remains 100%.

Implemented deterministic reviewed-annotation scoring and replay script, with hash/source/
claim/reference completeness checks and regressions. Annotations:
`data/stage-1.1/answer-reviewed-baseline-v1.json`. Scored cases and summary:
`docs/results/answer-reviewed-baseline-v1`. These are agent-reviewed development scores, not
independent human or held-out validation. No new thresholds or production-quality pass claimed.

Verification: 148 tests passed, two optional checks skipped; real cached retrieval models
executed. Ruff check/format and whitespace checks passed.

This requested baseline is complete; no document expansion or generation tuning performed.
The earlier blocked automated-evaluator status below is historical and does not block this
reviewed scoring route. No approval is required merely because earlier iterations failed.

---

# Current addition: semantic answer evaluation

The user authorized implementing faithfulness, answer correctness, semantic citation accuracy
and abstention accuracy over the existing five documents. The offline scorer, versioned reference
facts, calibration fixtures, CLI and regression tests are implemented. No new dependencies,
models, source documents or answer-generation changes are needed.

Evaluator validation: 8B passed 2/6 calibration fixtures with both rubrics. 14B passed 4/6
with the revised rubric: it correctly rejected wrong claims but still credited two missing
reference facts as covered. All three failed runs are archived under
`docs/results/answer-metrics-v1`; semantic scores are withheld. Deterministic abstention
accuracy is available independently: 20/20 on the saved 14B answers.

A concrete two-pass redesign is implemented: reference coverage is checked against generated
answer text alone, without the original reference/source excerpts. Both passes and their
rationales are preserved. A unit test proves reference-only text cannot enter the coverage
pass. It has not been evaluated with a live model. No further calibration run was made after
the three failures; the attached prompt requires a decision after unsuccessful iterations.
Regression: 141 tests passed, two optional checks skipped; real cached retrieval providers
executed. Ruff check/format and whitespace checks passed.
Next action: user decision on one focused two-pass calibration iteration using installed 14B.
No new model, dependency or document stage is proposed.

---

# Current work: answer phase 1.1 — five documents

2026-10-25: The user explicitly accepted retrieval/ranking as satisfactory and changed the
scope to local answer generation on the same five documents. The pending ranking-tuning
proposal below is superseded as the active task; no further ranking tuning is authorized or run.
The historical stricter ranking gate remains failed and its thresholds/labels are unchanged.

Implemented: strict local Ollama adapter; pinned installed Qwen3 14B/8B; bounded whole-chunk
context; structured cited claims; explicit insufficient-evidence responses; fail-closed output
validation; answer and answer-evaluation CLI commands. The answer index uses the existing
full-context BGE frozen retrieval profile, without parameter changes.

Default model: Qwen3 14B. Its revised-prompt development check passed 15/15 known-fact
questions, 5/5 correct abstentions, 100% exact citation provenance and zero invalid/provider
errors; generation median 9.67 seconds. Claims and quotes were reviewed.

Qwen3 8B also passed all 20 revised-prompt checks, with generation median 6.11 seconds.
Its separate config is available as the faster alternative. Both earlier failures are retained.

Completed: default answer CLI smoke passed with cited ownership claims. Full offline regression
with real cached embedding/reranking providers: 123 passed, two optional checks skipped (Docker
server parity and the historical strict ranking gate). Ruff check/format and whitespace checks
passed. No active blocker for the five-document answer implementation.

Reproduce: `HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false ERAG_MODEL_TESTS=1 uv run pytest -q`.
Use `uv run erag answer "Who owns AKS upgrades?"` for the default model, or add
`--answer-config config/answer-8b.yaml` for the faster alternative. Re-run a fresh answer check
with `uv run erag evaluate-answers --output artifacts/new-answer-check.json`.

Current evidence and compressed raw runs: `docs/results/answers-phase-1.1/summary.json`.
Next action: user review/discussion; remain at five documents until expansion is explicitly
approved. No narrative stage report generated. The comparison uses
identical retrieved evidence for both models and records generation-only latency. These are
small development checks, not new independent validation or proof of semantic entailment.

No additional source documents, model downloads, Python dependencies or cloud services added.
Next document stage remains unapproved. No new narrative stage report is being generated.

---

## Historical retrieval work

# Stage 1.1 — audited labels approved; quality gate still FAILED

Date: 2026-10-25. Exactly five original source documents. Stage 1.2 is not approved.

## Completed audit and approved label version

- User approved an independent agent audit, blind to labels, ranks, scores and targets.
- All 171 questions and 30 source units were audited; all quotes and IDs mechanically checked.
- Auditor: 149 answerable, 21 unanswerable, 1 partially answerable. Proposed differences for
  64 questions. Some proposed full grades contradicted their own required-fact scope.
- Primary source review isolated 15 conservative corrections across 14 questions. The other
  88 unit judgments remain unadopted and require adjudication.
- User explicitly approved those 15 corrections in a separate version and frozen re-evaluation.
- Approved labels: `data/stage-1.1/evaluations/source-audit-v1`. All original label files,
  question text, thresholds and five source documents remain unchanged.
- Audit/proposals/evidence: `docs/results/label-audit`. No narrative report generated.

## Re-evaluation result (no parameter changes)

Six previously frozen profiles were re-evaluated on 60 answerable questions plus five no-answer
cases. Four profiles executed actual retrieval; the window/fusion profiles reused verified
recorded rankings for exactly the same queries, corpus, chunks and parameters. Timing reuse is
explicit in their metadata. Nine audit/re-evaluation utility tests passed.

**None qualifies.** Descriptive best observed frozen profile: full-context BGE.
Hit@1 .9167 FAIL; Hit@3 .9833 PASS; Recall@5 1.0000 PASS; MRR .9478 FAIL;
nDCG@10 .9519 PASS. Threshold comparisons use unrounded values.
Latest rank-fusion profile: Hit@1 .8333, Hit@3 .9667, Recall@5 1.0000,
MRR .9019, nDCG@10 .9178. No failed profile was promoted; release defaults are unchanged.
These are observed-question results with an approved label revision, not new independent proof.

Results and profile provenance: `docs/results/source-audit-v1-reevaluation`.
Current status: `docs/results/current-quality.json`.

## Pending decision

The original three BGE tuning rounds are exhausted. Current approval covered auditing, the
15 versioned label corrections, and fixed-profile re-evaluation only; no new tuning was run.
A concrete proposal is in `config/next-ranking-plan.json`: up to three rounds using existing
BGE, focusing on standard acronym normalization, model-input metadata, and document context;
a blind agent would prepare prospective questions before tuning. No model downloads or new
source documents. Do not execute this proposal until the user approves it.

Original audit hashes and label lineage are preserved. Some audit proposals remain disputed;
do not silently adopt them, remove difficult cases, relax targets, or claim production readiness.
