# Phase 1 architecture

Current approved scope: twenty-document Stage 1.3, with retrieval and local cited answers
through one Python CLI. Five- and ten-document profiles remain reproducible. Chat, authentication,
cloud deployment and expansion beyond twenty documents remain future work.

| Module | Responsibility |
|---|---|
| `ingestion.py` | Five-source stage guard, UTF-8/content validation, stable source IDs and content versions |
| `chunking.py` | Heading/paragraph-aware word budgets, bounded overlap, source offsets and coverage validation |
| `models.py` / `types.py` | Replaceable embedding/reranking protocols and pinned CPU model adapters |
| `storage.py` | Qdrant local/server vector adapter, exact search and payload consistency checks |
| `bm25.py` | Deterministic lexical index over the same canonical chunk snapshot |
| `fusion.py` | Reciprocal rank fusion; raw scores never mixed |
| `pipeline.py` | Fresh index builds, atomic manifest publication, updates, logical deletion and retrieval |
| `evaluation.py` | Evidence-grounded labels, chunk/document metrics, per-query results, versions and timing |
| `config.py` | YAML configuration, type/range validation, unknown-key rejection and index fingerprints |
| `diagnostics.py` / `cli.py` | JSON event logging and inspection/evaluation commands |

A canonical snapshot supplies both indexes. A new Qdrant collection is built and checked against
chunk IDs and payloads before the active manifest is atomically replaced. BM25 reconstructs from
that manifest; readers never combine different chunk generations. Old collections remain available
for readers and are not automatically garbage-collected. A failed build leaves the old manifest
active. Logical deletion records source IDs in the manifest and rebuilds both active indexes;
`restore` clears tombstones. Physical source removal fails the five-document stage guard.

Document IDs are source-relative fixture stems, versions are SHA-256 of original UTF-8 content,
and chunk IDs are UUID5 of document ID/version/character offsets. Rebuilding unchanged input
retains logical IDs despite using a new physical Qdrant collection. A source/config/model change
requires rebuilding; startup rejects stale or inconsistent snapshots.

Dense search retrieves 20 candidates, lexical search retrieves up to 20 positive-score candidates,
and RRF k=60 merges their ranks. Hybrid retains the first 20 fused candidates. The optional
cross-encoder reranks only those candidates, never injects labelled positives. Exact Qdrant
search avoids approximate-index randomness for this small stage. Default retrieval mode is hybrid.

Diagnostics retain retrieved source text, metadata, initial component rankings, component score
names, fusion ordering, final ordering and timings. Models run on CPU with two threads and
fixed seeds, reject over-budget input instead of silently truncating it, and disallow remote code.

The embedded backend supports a single process per storage directory. Compose is prepared for
server access/concurrent clients. Server integration is verified using the pinned container and
Docker CLI; Compose CLI is absent, so the Compose invocation remains untested. This stage uses
whole-index rebuilds and has no writer coordination beyond atomic snapshot publication. Run
index mutations sequentially; distributed multi-writer behavior is not claimed.


## Local answer extension (2026-10-25)

```mermaid
flowchart LR
    Q[Question] --> R[BM25 + vector retrieval]
    R --> F[Weighted RRF + frozen BGE reranker]
    F --> C[Whole-chunk context with source IDs]
    C --> O[Local Ollama Qwen3]
    O --> V[Validate structured claims and evidence quotes]
    V --> A[Cited answer or insufficient evidence]
```

The answer profile copies the accepted frozen full-context BGE configuration without tuning
retrieval parameters. It uses a separate `artifacts/answer-index` so legacy retrieval indexes
and experiment defaults remain reproducible. Embeddings and reranking use the existing pinned
CPU adapters; Ollama controls local generation acceleration on the user's Apple M4 Pro.

| Module | Responsibility |
|---|---|
| `answer_config.py` | Strict local endpoint, model digest and generation/context budgets |
| `answering.py` | Replaceable generator protocol, context packing, schema/prompt, claim and quote validation |
| `ollama.py` | Standard-library HTTP adapter; installed-model digest verified before and after inference |
| `answer_evaluation.py` | Known-fact, abstention and citation-provenance checks with per-case evidence |

Generation uses temperature 0, seed 0, thinking disabled, structured JSON output, 8,192-token
context and at most 1,024 generated tokens. Whole ranked chunks are packed up to six sources,
then bounded conservatively using UTF-8 byte length plus output/format overhead. This is a
conservative tokenizer bound, not the model's exact token count. There is no silent text
truncation; overly large inputs are rejected or whole trailing sources omitted. These settings
improve repeatability but do not guarantee bit-identical output across runtime/hardware versions.

Question and source instructions are untrusted data in the user message. The system prompt
requires source-only claims, verbatim evidence and abstention when a material part is missing.
The generator cannot execute tools. The validator rejects unknown source IDs, non-verbatim
quotes, uncited claims and unsupported numeric strings; source offsets are derived from the
canonical chunk rather than trusted to model output. Valid citations are rendered by the
application. Invalid responses never become factual answers. These checks do not implement a
semantic entailment verifier; a quote can be authentic yet support the wrong claim, so factual
assessment remains distinct from citation provenance.

Only loopback HTTP is accepted. Redirects/proxies and automatic model pulls are disabled.
Truncated generations, digest drift and malformed responses produce explicit failures. The
existing installed models are Qwen3 14B and 8B, both Q4_K_M; their full identities live in the
answer YAMLs. No cloud API, new service, framework or Python dependency was added.


## Offline semantic answer metrics

`answer_metrics.py` scores saved answers without rebuilding retrieval or regenerating answers.
`erag score-answers` validates the original five-document corpus and the versioned reference
facts, runs evaluator calibration, then invokes the installed local judge through the same
Ollama adapter. It preserves atomic judgments, citation judgments, required-fact coverage,
model identity, rubric/schema, inputs/hashes, calibration results and errors. No dependencies,
model downloads or document expansion are introduced.

- Faithfulness: supported atomic answer assertions / all atomic answer assertions, judged
  against retrieved context only. Answers without factual claims are excluded from this ratio.
- Answer correctness: mean F1 over the 15 answerable questions, combining factual precision
  (correct atomic assertions / assertions) and required-fact recall. Abstentions and generation
  errors on answerable questions receive zero. Correctness uses verified reference evidence
  separately from the retrieved context.
- Citation accuracy: semantically supporting citations with valid original-source provenance /
  all citations. Each citation must support its whole associated claim; an authentic but
  unrelated quote fails. Unanswered questions are excluded from this citation ratio.
- Abstention accuracy: correct answered/insufficient-evidence decisions / all 20 questions.
  False abstentions and invalid/provider-error responses are incorrect decisions. Unsupported-
  question abstention recall is also recorded separately.

These are descriptive metrics, with no newly inferred quality thresholds. Semantic estimates
are withheld if any judging error leaves incomplete scoring. Absence of answered claims makes
faithfulness/citation accuracy undefined rather than 100%. Six positive/negative calibration
fixtures exercise wrong values, swapped owners, irrelevant authentic quotations, omitted facts,
and requested versus observed state. Calibration failure blocks scoring. Passing six fixtures
would still not prove human agreement, unbiased decomposition or independent reliability.
Same-family and self-judging biases must remain explicit; the original generation results are
not changed by evaluating them.


The proposed third rubric uses two passes: source support/correctness against source evidence,
then required-reference assertions against the generated answer text as the only evidence.
Raw judgments from both passes are retained; only the second pass supplies reference coverage.
This prevents source text from entering the answer-coverage context. The design is unit-tested
but live calibration is pending. Three earlier single-pass calibrations failed (8B: 2/6 twice;
14B: 4/6). No aggregate semantic scores have been published. Deterministic abstention remains
available even when semantic calibration fails. Calibration artifacts are development sanity
checks, not evidence of general evaluator reliability.


## Reviewed baseline scoring

`reviewed_metrics.py` and `scripts/score_reviewed_answers.py` calculate the same four metrics
from explicit reviewed annotations. No generator or judge provider is invoked. Review metadata
is labelled agent-reviewed; every annotated answer and reference is bound to frozen file hashes,
source corpus versions and exact claim text. Every question must have exactly one annotation;
answered cases require valid judgments for every original claim/citation and required fact.
Atomic assertions and semantic citation reasons remain inspectable. Citation provenance is
rechecked against original sources before calculating support. Existing scoring formulas are
reused, so reviewed values can later be compared with automated evaluator outputs.

The initial reviewed baseline flags one requested-versus-actual validity overstatement. Its
strict qualifier interpretation is recorded, not treated as independent human ground truth.
The user removed the unsuccessful-iterations approval gate; its earlier pending-decision notes
are historical. The requested baseline uses direct review rather than further local judging.


## Stage 1.2 isolation and validation repair

`Config.stage` explicitly selects approved identifiers for 1.1 (five) or 1.2 (ten); file count
never grants permission automatically. Stage 1.1 index fingerprints are preserved; new-stage
fingerprints include the stage. Original source versions/chunk IDs remain stable across the
expansion. Canonical payload checks, logical deletion, restoration and stale-index rejection
apply to all ten sources. Each stage uses separate profiles and Qdrant state directories.

Only the 1.2 answer profile enables `validation_retries: 1` and `align_quote_fragments: true`.
Validation may map sentence fragments to a unique original-source location with an initial-case
or final-punctuation variation. Internal words, punctuation and numeric identifiers are unchanged;
ambiguous matches fail. The application returns the literal source span and records the generated
quote and alignment rule. This establishes provenance, not semantic entailment.

After a remaining validation failure, at most one request includes the invalid claims and validator
feedback, the original question and unchanged source excerpts. No expected answer is supplied.
Every retry is validated again; excessive context and persistent failures remain rejected.
Initial valid answers are not retried. Per-attempt usage, latency and failures are retained;
evaluation diagnostics retain raw generations. Generation time includes all actual attempts.
The initial schema's duplicate textual copy is omitted on repair to fit feedback without dropping
source chunks; the structured-output schema is still provided to Ollama.

Reviewed 1.2 metrics reuse the existing formulas and hash-bound annotation scorer. The final
validation replay retains actual measured first-pass generations and makes fresh repair requests
only where needed. It is explicitly marked as replay and is separate from the fresh CLI smoke.


## Stage 1.3 expansion

Stage 1.3 explicitly allows the twenty approved source IDs and retains all ten prior documents
and canonical chunk IDs. The same index/payload/fingerprint and deletion/restore checks apply.
New source/index profiles keep previous states isolated. Candidate count remains twenty;
chunking, embeddings, weighted RRF and BGE parameters are unchanged. Generation remains local
14B, with the new profile context budget 12,288 to accommodate six relevant source chunks and
repair feedback. Output budget and the one-retry/unique-fragment alignment policy are unchanged.
No old profile is modified. Stage 1.4 and unauthorized source sets remain rejected.

The frozen evaluation compares all six strategies, with fresh local answer generation and a
separate fresh CLI smoke. Retrieval diagnostics are reused from the real ranking evaluation,
with explicit timing scope. Thirty supported and six unsupported answers are source-reviewed;
atomic/citation judgments, reference coverage, grouped results and conservative ambiguities are
preserved. Factual precision penalizes unsupported additions even when all requested facts are
present. A reference-overconstraint note is retained without silently rewriting the gold facts.


## First real-document track

`real-1` is a separate, explicitly approved five-page pilot, not synthetic stage 1.4. Its fixed
source IDs correspond to the first five Kubernetes URLs in the supplied manifest. Ingestion
requires complete verified acquisition metadata and exact frozen text hashes, supplies publisher
URLs as source identifiers and tags records as real-world-stage1-v1. It cannot ingest the other
25 manifest URLs or bypass synthetic-stage boundaries.

`article_extraction.py` selects only Kubernetes td-content and tracks explicit div boundaries to
survive optional closing tags. Headings, preformatted code indentation, definition lists and table
rows survive; scripts/navigation/feedback are excluded. Raw sources plus HTTP/title/hash/version
metadata allow inspection and replay. Initial truncation was detected and rejected before indexing.
No generic fallback publisher parser or new dependency is installed.

Real retrieval uses the fixed existing profile and 14B generation. Source-reviewed selected-span
labels are not an exhaustive relevance pool. Standalone queries do not include user topic headings;
that limitation and partial-answer/controller scope ambiguities are explicit. Provisional reviewed
scores reuse the existing formulas and do not imply human signoff. Software-test success and exact
quote provenance are separate from factual/semantic quality. All full source copies stay in local
artifacts. The first five stay fixed pending review; expansion needs approval.


## Corrected real-pilot retrieval and configurable answer providers

The v2 real profile retains all five frozen sources and canonical chunk IDs. Retrieval-only
representations prepend the document/heading path; this changes the index fingerprint and uses
an isolated index-v2 directory. Source text and citation offsets remain canonical. Independently
configurable branch depth60 and rerank depth40, with balanced RRF, prevent the prior lexical
candidate starvation. Link-only final sections remain in the corpus but are excluded from retrieval.

Answer context expands anchors to exact neighboring source spans and merges overlapping ranges
without concatenating duplicated chunk text. The answer fact catalog preserves adjacent conditions,
paragraph headings and long table rows. Existing BGE reranks catalog facts; local 14B chooses bounded
evidence IDs. Code renders the selected original facts with exact-span citations. This extractive
mode prevents rewritten definitions and fabricated quotes, but does not prove question coverage
or relevance. Expected-fact patterns remain distinct from semantic correctness.

`create_generator` selects Ollama or the optional installed Codex CLI using the answer config's
`provider` field. Existing configs default to Ollama. Both providers share grounding/validation;
there is no automatic fallback. Codex uses isolated ephemeral/read-only execution, structured
stdout/final output, disabled shell/search/apps and ignores user configuration. Tool events,
timeouts and malformed/incomplete outputs fail closed. No CLI authentication files are modified
by the adapter; normal CLI authentication is required when the provider is selected. Model names
for local and Codex remain separate so toggling the provider does not overwrite the local pin.
Codex transport passed a live constant-JSON smoke test without document content. Document
quality evaluation remains local 14B.


Source-fact extraction derives the initial Markdown-fence state from the original document's
prefix at the excerpt's absolute start. It must not assume that an excerpt begins outside code:
a closing fence at the start of a partial YAML example otherwise hides all following prose.
The derived flag is answer-context metadata, not a mutation of indexed chunks or source text.

Optional coverage review is bounded and additive. Primary facts are validated before review;
new IDs must refer to supplied catalog facts and fit the six-claim limit. Empty additions retain
the valid primary answer. Unknown IDs/model changes fail closed; original abstentions are not
converted into factual answers by this pass. Local review uses a separate bounded non-thinking
request by default, retaining the same pinned14B weights; primary reasoning remains configurable.
The capture wrapper records both calls. This check can improve missing conditions but is not
an independently calibrated semantic judge, and unrelated source background may remain.

Evidence evaluation consumes saved rankings, exact context sources, the saved fact catalog and
final citations. Source/version/offset checks bind all witnesses to the same frozen corpus.
Raw top-k chunk metrics stay unchanged. A separate, explicitly versioned relevance proposal
allows legitimate alternative passages; completeness counts required facts rather than chunks.
Each fact can have OR alternatives, each containing AND source spans; coverage merges touching
source intervals without bridging missing text. Expanded context, model-visible catalog and
selected-quote coverage are separate stages. No-requirement controls contribute no fact-recall
denominator. A missing exact witness is not a semantic incorrectness judgment.


Answer repairs (real-five v10): canonical chunk ranking remains unchanged. A bounded second
search derives prose facts from the first two distinct retrieved documents, ranks with the same
BGE model, and reserves section/lexical coverage. Primary selection retains focused chunk facts;
review adds source IDs without discarding validated primary evidence. Strict source supersets
may replace clipped duplicates. An initial abstention uses a fresh combined-catalog selection;
ordinary coverage reuses the primary prefix. Both paths reserve their full review overhead.
All context and provider failure phases are captured; the user question is last in the payload.

Ollama can recover once from thinking timeout/token exhaustion using the same pinned model,
same messages/schema and a bounded non-thinking request. Pin changes, HTTP/protocol/tool/JSON
errors do not trigger that recovery. Initial failure, both elapsed costs and request count are
recorded; there is no cloud fallback.

`section_context_completion` optionally completes selected extractive quotes with exact bounded
Markdown section supersets (<=2000 chars), preferring small parent policy sections; large parents
use closed subsection prefixes. Whole paragraphs repair clipped prose. Fenced examples are
excluded. Duplicated/contained spans are removed without losing original selected content.
Original model claims/source IDs/catalog remain saved. New SC sources are post-generation
context, excluded from expanded/model-visible coverage. The step has no query/reference/model
inputs and no effect on model prompts. CPU replay of recorded generations is explicit, preserves
parent hashes and validates every source citation and original-assertion superset.


Brief synthesis follows validated evidence selection and section completion. It sees only the
source excerpts associated with selected evidence, the original question and a strict cited-claim
schema. At most3 claims and the configured word limit are enforced without truncation. Every
quote is rebound to original document/version offsets. A separate same-provider review checks
claim support, requested-part completeness and focus. One rejected semantic candidate may be
revised once, then reviewed again; malformed output, invented quotes and changed models fail
closed. Full evidence remains in separate JSON fields and is available through --detail.
Same-source citation markers are deduplicated for display while individual exact quotes remain.
The review is an operational check, not calibrated independent evaluation or a quality score.
