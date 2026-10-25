# Decisions

2026-10-24: Repository inspected; no existing source, tests, configuration or AGENTS.md found.
No implementation reused. Stage 1.1 is the only approved document stage.

The initial proposal follows the supplied stack. Automatic approval review initially rejected
dependency resolution under the major-dependency gate. The user subsequently approved the
listed Python packages and model downloads, which are now installed and locked.

Choose CPU-friendly retrieval models and immutable revisions, without claiming superiority
before comparison. Sentence Transformers documents separate bi-encoder retrieval and
cross-encoder reranking. MS MARCO cross-encoder scores are logits, not probabilities.
References consulted:
- https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- https://sbert.net/docs/cross_encoder/usage/usage.html

Evaluate all strategies independently; never add raw scores from different scoring systems.
Use evidence spans to label chunks across chunking configurations, and retain exact derived
chunk IDs in experiment reports. Label changes must correct factual errors, never inflate metrics.
No-answer cases are reported separately because their recall/MRR/nDCG are undefined.

## Approvals and execution

- User explicitly approved the proposed retrieval/ranking thresholds and three tuning iterations.
- After clarification that local models perform embedding/reranking (not chunking or generation),
  user said "then ok go ahead", approving the listed Python packages and model downloads.
- Python 3.13.13, packages and model revisions are locked and actually installed/downloaded.
- Official Qdrant Docker image download was separately rejected by automatic approval review.
  The user subsequently explicitly approved the image and local tests; download/run completed.
- Qdrant v1.20.0 is proposed for Compose, matching client version; official release reference:
  https://github.com/qdrant/qdrant/releases/tag/v1.20.0 . The approved image digest is now pinned.

## Measured selection and limitations

Three preregistered tuning runs compared 100/140/180 words at 20 overlap. BM25 at 140 words
won the predefined rule; hybrid remains the CLI default as requested. Do not change that
selection based on held-out results. Reranking helps recall but can hurt graded ranking.
No-answer ranking metrics are undefined; return and report candidates without claiming abstention.

Use logical deletion plus rebuild to satisfy active-index deletion while retaining the fixed
five-source dataset. Retain old collections for snapshot readers; do not claim physical erasure.
Embedded storage is single-process, and index writers must be serialized. No multi-writer claim.
Post-evaluation fixes addressed error reporting, a deprecated model property, embedded search
warnings and a git metadata diagnostic; ranking behavior was replayed on tuning data only.

## Server validation completed

- Official Qdrant v1.20.0 image pinned to digest
  `sha256:9bfc89f9914fc14876a88c05270685921a55b8877b254a98bcc1cc2f0a424cd2`.
- Full test suite: 53 passed, zero failures/skips. Real-model server check confirms six variants,
  embedded/server ranking parity, idempotent rebuild, update, logical deletion and restore.
- An extra smoke paraphrase places the expected source second under BM25 on both backends;
  reranking places it first. The initial top-one assumption failed and was retained as a finding.
  Backend verification checks identical rankings and the agreed top-three relevance cutoff.
  No official labels or acceptance thresholds changed.
- This machine has no Compose CLI. Automatic approval review rejected downloading an extra
  Compose executable; the suggested existing-Docker alternative was used successfully.
  Compose invocation remains untested; equivalent server settings were exercised with Docker CLI.
- The temporary verification container was stopped/removed after recording results. No additional
  document stage was started. The Stage 1.1 gate passed; Stage 1.2 awaits explicit approval.

## Stricter target improvement cycle

User required Hit@1 >= .95, Hit@3 >= .98, Recall@5 >= .95, MRR >= .95 and nDCG@10 >= .95.
These supersede the earlier weaker chunk gates. Missing metrics fail closed. Original results
remain historical and do not certify the new targets.

Original observed questions were retired to development. Fresh v2 and v3 validations failed
some targets and were preserved. The user explicitly approved BAAI/bge-reranker-v2-m3
(~2.27 GB weights), pinned commit 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e, and three
further tuning rounds. Whole-chunk BGE passed development but failed Hit@1 on validation v3.
Two-sentence BGE passed development but failed four targets on 60 new phrasings (validation v4).
No failed profile was promoted.

Final approved BGE round compares rank-only combination of whole-chunk, evidence-window and
initial retrieval views. Selection is constrained to 90 whitelisted development IDs and their
frozen label hash. After automatic review questioned reading a retired-validation archive,
whole-context scores were recomputed directly from the development file; no validation-result
archive is used for weight selection. A later v4 check is labelled regression replay.

Original evidence, relevance grades, source documents and thresholds were not changed.
Repeated phrasings of fixed corpus facts are not an external or intent-disjoint benchmark.
The quality test is opt-in via ERAG_QUALITY_TESTS=1, prints the five target/result/pass rows
at the end of pytest and fails the suite on any unmet gate.


## 2026-10-25: User-authorized local answer scope

The user accepted ranking as satisfactory and explicitly requested answer generation on the
same five documents, offering installed local Qwen models. This supersedes the original
retrieval-only exclusion for answer generation; it does not authorize additional documents,
ranking tuning, cloud services or chat. The strict historical ranking gate remains unchanged.

Use the existing full-context BGE profile for answers, with a separate state directory. Avoid
framework/dependency additions: a replaceable generator protocol and standard-library Ollama
HTTP adapter are sufficient. Inspecting the local server found Qwen3 14B and Qwen3 8B (rather
than 7B), Q4_K_M, on an Apple M4 Pro with 24 GB RAM, Ollama 0.40.7. Full model digests are
pinned in the answer YAMLs; no new model downloads are required. Begin with 14B and compare
8B using fixed questions and identical evidence before judging quality/speed.

Require structured claims with exact source quotes and explicit abstention. Validation fails
closed, but exact quotation/numeric checks are not a semantic entailment proof. Keep factual
answer checks, citation provenance and unsupported-question behavior separately measurable.
The small fixed answer set is development evidence; no claim of independent held-out or
production quality follows from passing it. No narrative stage report was created, per user.


Both installed models passed the revised-prompt development checks (15 factual, five abstention),
with generation medians 9.67 s for 14B and 6.11 s for 8B. Retain 14B as the initial default;
8B remains selectable for lower latency. Equal scores on 20 observed questions are insufficient
to conclude that general model quality is identical. Full real-provider software regression:
123 passed, two optional integrations/gates skipped. Passing answer checks do not change the
historical retrieval thresholds or authorize a new document stage.


## Semantic metrics extension

User requested implementation of all four answer metrics. Reuse installed Ollama models,
without new providers/dependencies. Keep semantic citation support separate from exact quote
provenance and score reference completeness separately from claim precision. Version explicit
reference facts without modifying original questions/evidence/answerability labels. Gate semantic
scores with positive/negative evaluator sanity checks; do not publish values after calibration
failure. The first three single-pass calibrations failed (8B twice, then 14B); actual raw judgments
are preserved, not relabelled. A two-pass answer-only coverage redesign is implemented and
unit-tested; request the user's decision before the next local calibration iteration as required
by the attached unsuccessful-iterations gate. Deterministic abstention is 20/20 independently.
No numerical acceptance thresholds, new documents or answer-generation changes were introduced.


## Reviewed answer baseline and user steering

The user explicitly revoked the unsuccessful-iterations approval requirement, chose 14B-only
answer generation for now, and authorized the reviewed baseline. Review all saved 14B answers
directly against original sources and reference facts, label them agent-reviewed (not human or
independent), and calculate reproducible scores without a local judge. The reviewed scorer
requires frozen input hashes, exact claim/reference matching and complete annotations.

Faithfulness 29/30; semantic citation support 28/29; correctness macro F1 14.5/15; abstention
20/20. One conservative qualifier finding distinguishes requested from actual certificate
validity. Retain that reviewable judgment and original outputs; do not silently rewrite the
answer, reference facts or source to raise the score. Raw local-judge calibration failures
remain historical evidence; no new model calls were needed for this baseline.


## Phase 1.2 authorization and experiment plan

User explicitly authorized the ten-document stage. Preserve the original five source bytes and
chunk identifiers, and keep stage-specific configurations/state so old baselines remain replayable.
Make the ingestion stage explicit rather than inferring permission from file count. Reject
unapproved stages and document IDs. Reuse pinned embeddings, BGE reranker and Qwen3 14B;
no new package, provider or model download. Keep the prior accepted ranking parameters fixed
before opening any new evaluation results.

Add five adjacent runbooks with meaningful overlap and scope boundaries. Stage 1.2 questions
must account for newly answerable PostgreSQL/Redis questions and legitimate alternative evidence
introduced by new runbooks. Review these against sources before any ranking outputs are seen.
Freeze 36 answer questions and two retrieval splits; distinguish 15 legacy regression questions
from fresh stage questions. No model selection or tuning on the heldout split. Splits/labels are
agent-authored and synthetic, not independent human validation. Use the same transparent reviewed
semantic formulas; do not trust local judges that failed calibration. Preserve individual
failures, ranking metrics, answer evidence, reviewed annotations, hashes and runtime versions.


Stage 1.2 execution retained the fixed hybrid/BGE profile even though vector/BGE descriptively
had slightly higher heldout recall; do not use heldout results for selection. All existing ranking
targets passed. Two genuine output-validation failures motivated bounded repair and conservative
source-fragment alignment, enabled only in the new answer profile. Exact original evidence,
failed attempts and per-attempt diagnostics are preserved; unambiguous punctuation/case alignment
does not relax lexical/numeric checks or establish semantic support. Record final scoring as
measured first-pass replay plus actual repairs, with a separate fresh CLI smoke, rather than
claiming all outputs were freshly regenerated under the revised validation profile.

Reviewed metrics retain the earlier strict qualifier and whole-claim citation rubric. Explicitly
record partial reference-scope coverage in new-26 without rewriting expected facts after seeing
answers. All next-stage documents remain unapproved. Legacy defaults/profiles remain available;
use the explicit stage-1.2 profile for the current ten-source corpus.
