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


## Phase 1.3 scope and preregistered profile

User authorized twenty documents, retaining all prior ten. Extend the explicit stage ID set to
1.3; reject 1.4 and later sets. Keep ranking parameters, 20-candidate pool, chunking and model
revisions fixed before evaluation. Raise only the new answer profile's context budget from
8,192 to 12,288 to fit broader evidence/repair context; retain 14B, output limit, seed, temperature,
quote alignment and one-retry policy. No old profile changes or new dependencies/downloads.

Prepare ten new runbooks and freeze thirty supported questions (ten regressions, twenty fresh)
and six unsupported questions before retrieval. Reference facts are atomic and limited to the
question's requested information to reduce earlier compound-reference overconstraint. Source-
review selected regression alternatives before ranks; no new complete-answer alternatives were
introduced for those selected questions. Compare all six retrieval strategies without selecting
on heldout. Keep local model judges excluded from semantic scoring; annotate actual answers
against sources and required facts with reviewable reasons. Record all failures and uncertainties.


Phase 1.3 execution: the unchanged hybrid/BGE profile passed all ranking targets on the fifteen
supported heldout cases. Every answer was freshly generated with 14B; one bounded ownership
retry succeeded. Do not confuse the thirty passed fact-pattern cases with semantic correctness:
source review found unsupported qualifier/storage-scope additions and one reference-overconstraint
caveat. Keep original outputs/sources/facts and record explicit adjudication reasons. Separate
legacy/new question groups so larger aggregates do not hide retained findings. Exact source
provenance remains distinct from semantic citation support. No further generation/ranking tuning
or next-stage expansion is implied by completion. Prior stored five/ten-source indexes still
validate under the extended configuration guard.


## Real-document validation authorization

User accepted the proposed confidence step: real documents and independently phrased questions.
Inspect the workspace before requesting files; no selected real enterprise documents found.
Prepare source/question hashing without pretending to ingest or evaluate unknown files. Request
selected folder/URLs and natural team questions. Retain the existing fourteen-billion-parameter
local model; no paid provider, connector or parser installation is implied in advance. Do not
substitute public/synthetic documents or agent-generated questions for the intended real-world
validation. Keep this pilot separate from versioned synthetic stages and capped at twenty selected
sources until scope changes explicitly. Preparation tests passed; actual data/metrics pending.


## Official-document pilot execution

The supplied actual folder `data/real-world/stage1` contains a URL manifest and first-five-only
instructions, superseding the generic preparation cap for actual ingestion. Verify the five
canonical official pages, record title plural/singular differences, and stop at five. Reject
truncated extraction before any model use; preserve raw sources and metadata, then re-extract
complete main content with code indentation/definition/table handling. Keep large publisher
copies/detailed results in ignored artifacts. Add only a manifest/hash-bound real-1 ingestion
branch with explicit five IDs and dataset/URL provenance; synthetic configurations remain intact.

Freeze the user's twenty question strings before outputs and mark authorship as user-supplied,
not independently verified human authorship. Supplement with two explicitly agent-authored
live-state controls. The baseline used standalone queries, omitting supplied topic headers; record
this input limitation rather than silently adding scope after seeing results. Source-grounded
evidence/labels were independently constructed from the frozen corpus before model outputs.
They are provisional agent judgments. Record scope ambiguities and selected-span non-exhaustivity
rather than claiming certified true-relevance accuracy or rewriting labels to raise results.

Initial real-corpus quality failed despite passing software tests and correct source acquisition.
Document discovery is stronger than selected-evidence chunk recall; incomplete evidence,
unconditional/storage/type misstatements and citation misbinding remain visible. Compare to
synthetic results descriptively, not as a controlled paired causal experiment. Do not tune on
these observed questions and call their replay independent validation. Human adjudication and
separate development questions are next; no ten-page expansion authorized.


## 2026-10-26: Real-pilot fixes and optional Codex answers

User authorized repairing the observed real-document failures and adding a single answer-provider
config switch. Keep synthetic profiles and original real baseline intact; all observed-question
replays are development results. Use balanced RRF, richer retrieval representations, independent
candidate depth and bounded original-source context. Do not adopt new labels or thresholds to make
the metrics pass. The ranking quality gate remains separate from answer expected-fact checks.

Tested a source-fact selector to prevent quote/definition rewriting. Early variants falsely
abstained and are retained as failed experiments. Heading/paragraph preservation, table-row
handling, fact-level reranking and a bounded ID-only selection contract improve coverage. The
extractive output may be verbose and still requires semantic review; a matching regex is not an
answer-correctness judgment. The thinking option is explicit and local-only, with a latency cost.

Answer `provider: codex` invokes the installed CLI through the existing generator protocol; it
is opt-in and sends retrieved evidence to Codex when used. Local14B stays default. Codex model
selection is independent of the pinned local model. There is no new Python dependency, automatic
model download or silent cloud fallback. Tests simulate the CLI subprocess; a live constant-JSON transport smoke also passed with no
document content sent. Official noninteractive/schema interface was checked alongside the installed CLI's
help: https://developers.openai.com/codex/noninteractive/ .


A full unbounded-thinking coverage experiment timed out. Keep it as a failed experiment; use
bounded non-thinking coverage on the same pinned model instead. A replacement-style review
removed relevant toleration facts, so coverage is additive and cannot discard validated primary
facts. Found and fixed excerpt fence inversion: code-tail excerpts were hiding the actual prose
needed for cross-namespace routing. This was an answer-catalog bug, not missing corpus content.

2026-10-26: Keep the real-document pilot at five articles. Separate ranking relevance from
required-fact availability and final-answer completeness. Store source-reviewed development
alternatives as an unadopted v2 proposal; retain all original labels and metrics. Freeze the
existing pipeline against its pre-question source archive, then run the20 new user question
wordings once. Preserve user topic context and the explicit steady-state q11 scope. Concepts
overlap development questions; labels are agent-reviewed, and two unsupported controls are
reused, so do not claim independent human or wholly disjoint validation. No post-run tuning.


2026-10-26, authorized real-five repairs: keep both observed sets as development data. Retain
all initial results and failed/interrupted probes; do not reuse the fresh-validation label for
new tuning. Keep local 14B/BGE/MiniLM pins and the same five documents. Restore thinking primary
selection after non-thinking probes lost useful facts; use bounded recorded recovery for actual
timeout/exhaustion. Fresh combined selection handles abstention rescue; prompt layout places
catalog first and question last. Full 43-case inference completed before integrating deterministic
section completion, so its postflight input checks remain meaningful.

Use document structure to retain defaults, alternatives and complete paragraph context in final
quotes. This is a bounded source-only renderer, not query/reference-conditioned answer patching.
Record model-visible versus post-generation sources separately. Reuse actual recorded selections
for this deterministic step and disclose that reuse; verify identical model inputs under the flag.
Do not claim fresh end-to-end timing, independent validation or a passing initial-ranking gate.


2026-10-27: User requested Codex answer generation. Use a dedicated Codex profile as the answer
CLI default, preserving the exact local v10 profile for reproducible benchmarks. Both profiles
share the same five-document retrieval and source validation. Explicit evaluation profiles and
historical local scores are unchanged. A real PostgreSQL question passed with Codex; do not
infer aggregate quality or a latency improvement from one smoke test.


2026-10-27, blank CLI response: structured provider failures must never print a blank successful
answer. Keep JSON diagnostics intact, show plain errors on stderr and return2 for provider and
validation failures. Pin the existing verified Codex executable in the runtime profile to remove
terminal PATH ambiguity; do not change CLI isolation or introduce a fallback provider. A live
503 question passed, but the user's earlier provider cause was not recorded and remains unknown.


2026-10-27: User authorized brief output. Make brief cited synthesis default in the active
Codex profile, preserving historical local/extractive configs. Do not shorten by cutting text
or hiding missing qualifications. Retain full evidence separately and offer --detail. Hard-bound
claim count and words; review factual support, completeness and relevance before exposing claims.
A single semantic revision is allowed, with rejected diagnostics preserved; there is no fallback
to long output after failure. Live ClusterIP answer is41 words, PVC-retention answer29 words
and preserves default Retain plus the whenDeleted Delete exception. Earlier source-copying
metrics are not applicable to new synthesis; broad evaluation remains separate.
