# Current stage: Phase 1.2 — ten documents

Ten synthetic DevOps runbooks are indexed in 52 chunks. The original five source files and
chunk IDs are retained unchanged. Stage-specific profiles preserve Phase 1.1 replay:

```sh
HF_HUB_OFFLINE=1 uv run erag --config config/stage-1.2.yaml index
HF_HUB_OFFLINE=1 uv run erag answer 'What are the required steps for a planned production Redis failover?' --answer-config config/answer-stage-1.2.yaml
```

Generation uses the same pinned local Qwen3 14B. Phase 1.2 enables one bounded evidence-
validation repair, and conservative quote-fragment alignment for unique literal source spans
with only initial-case/terminal-punctuation differences. Generated and aligned quotes are
recorded. Word/value changes and ambiguous alignments remain rejected. Legacy answer profiles
retain strict matching and zero automatic retries.

The frozen hybrid+BGE profile meets all existing ranking targets on 12 answerable heldout
questions: Hit@1/Hit@3 100%, Recall@5 97.92%, MRR 1.0, nDCG@10 0.9860. All six retrieval
variants were compared, with no ranking tuning or model selection on the heldout results.
The full 36-question answer check covers 30 supported questions and six correct abstentions.
Agent-reviewed baseline: faithfulness 98.63% (72/73 assertions), correctness mean F1 97.22%,
semantic citation accuracy 97.14% (68/70 links), abstention accuracy 100% (36/36).

All 36 first-pass model outputs were generated locally; two were initially rejected. Final
validation reuses recorded first attempts and makes only necessary fresh repairs. Ranking/
generation timing reuse is explicit; replay elapsed time is not a fresh end-to-end benchmark.
A fresh ten-document CLI generation also passed. Raw initial/final outputs, failed repair
attempts, review findings, source snapshots, hashes and test evidence are preserved under
`docs/results/stage-1.2`. The reviewed baseline remains a small synthetic development check,
not independent human validation or production certification.

```sh
# Replay reviewed metrics without a local model call:
uv run python scripts/score_reviewed_answers.py --answers docs/results/stage-1.2/answers.json.gz --references data/stage-1.2/answer-semantic-references-v1.json --reviews data/stage-1.2/answer-reviewed-baseline-v1.json --output artifacts/my-stage12-reviewed.json
# Run a fresh evaluation of the current profile; preserve earlier runs:
HF_HUB_OFFLINE=1 uv run python scripts/evaluate_stage12.py --output artifacts/stage-1.2/replay-001
HF_HUB_OFFLINE=1 ERAG_MODEL_TESTS=1 uv run pytest -q
```

Verification: 167 tests passed, two optional checks skipped. Stage 1.3 (20 documents) remains
unapproved. Historical Phase 1.1 results and commands follow below.

---

# Agent-reviewed answer baseline

For the saved Qwen3 14B answers on the same five documents, direct source review produced:

| Metric | Baseline | Basis |
|---|---:|---|
| Faithfulness | 96.67% | 29/30 atomic assertions supported |
| Answer correctness | 96.67% | Mean F1; 14 complete answers and one 0.5 partial answer out of 15 |
| Citation accuracy | 96.55% | 28/29 citations support their whole associated claim |
| Abstention accuracy | 100% | 20/20 answer-or-abstain decisions; all five unsupported questions declined |

The only flagged claim asserts actual 90-day certificate validity where the source establishes a
requested duration. The source also instructs checking actual certificate dates. This strict
qualifier interpretation is recorded for review. No original answer, source or expected fact
was changed to calculate these scores. Exact quote provenance is still 100%; that is distinct
from semantic citation support.

Annotations: `data/stage-1.1/answer-reviewed-baseline-v1.json`. Summary and compressed scored
cases: `docs/results/answer-reviewed-baseline-v1`. This is an agent-reviewed development
baseline, not independent human validation. Replay needs no model calls or active index:

```sh
uv run python scripts/score_reviewed_answers.py --output artifacts/reviewed-replay.json
```

The scorer checks original source/corpus versions, frozen input hashes, complete case coverage,
claim text, reference facts, judgment structure and citation offsets. Source support, factual
precision/completeness, citation support and abstention remain separately measurable. Local
judge failures described below remain historical; they do not prevent reviewed scoring.

---

# Semantic answer metrics

Faithfulness, answer correctness, semantic citation accuracy and abstention accuracy are
implemented in `erag score-answers`. This is offline scoring of saved answers. It does not
regenerate answers or rebuild retrieval. Source-verified reference facts retain all original
questions, source evidence and answerability labels.

**Automated-judge status:** abstention accuracy is 20/20 on the saved 14B development answers.
Automated semantic scores are unscored: both installed local evaluators failed sanity checks.
Failed calibration is preserved in `docs/results/answer-metrics-v1`; no unreliable scores or
semantic-quality pass is claimed. A separate coverage pass is implemented and tested but its
next local calibration run has not been executed. The user subsequently removed the
unsuccessful-iterations permission requirement and requested the reviewed baseline above.

To try the automated local judge separately:

```sh
uv run erag score-answers --answers docs/results/answers-phase-1.1/v2-qwen3-14b.json.gz --output artifacts/my-semantic-check.json
```

The default local judge is pinned Qwen3 14B (`config/answer-judge.yaml`). The 8B judge config
is retained for replay of failed checks. Both models are already installed. A failed calibration
still writes deterministic abstention results, with semantic scores explicitly unscored, and
exits 2. A judging error also withholds aggregate semantic scores. Results are estimates, not
human gold labels; same-family/self-judge biases and the small development set remain limitations.
See `docs/architecture.md` for exact metric definitions.

---

# Five-document answers

The user accepted the existing retrieval/ranking as satisfactory and authorized local answer
generation on the same five documents. Document expansion remains a separate decision.
The old stricter retrieval gate below remains a historical failure; its targets and labels were
not changed to authorize this answer extension.

The answer pipeline uses the frozen full-context BGE retrieval profile in
`config/answer-retrieval.yaml`, then Qwen3 through the existing local Ollama server.
`config/answer.yaml` selects the installed **Qwen3 14B**; `config/answer-8b.yaml` selects the
installed **Qwen3 8B**. Each model is pinned by its complete Ollama digest. No new model or
Python package is downloaded by the answer command.

Run from the repository root with Ollama running at `http://127.0.0.1:11434`:

```sh
uv sync --locked
HF_HUB_OFFLINE=1 uv run erag --config config/answer-retrieval.yaml index
HF_HUB_OFFLINE=1 uv run erag answer 'What is the IP address of pe-ledger-prod-01?'
HF_HUB_OFFLINE=1 uv run erag answer 'What is the database password?' --json
HF_HUB_OFFLINE=1 uv run erag answer 'What triggers CertificateExpirySoon?' --answer-config config/answer-8b.yaml
```

Answers contain source references. JSON output includes each claim's exact evidence quote,
original document offsets/version, chunk ID, retrieved sources and model/runtime identity.
Unsupported questions return `insufficient_evidence`. Malformed output, invented numeric values
or invalid quotations fail closed as `invalid_response` (exit 2); provider/config errors exit 1.
Quote provenance validates where evidence came from; it does **not** prove every claim is
semantically supported. Inspect the answer checks and evidence before relying on factual claims.

The fixed initial answer check contains 15 answerable questions and five unsupported/partial/
adversarial questions. It is a small development check, not independent held-out evidence or
production certification. Re-run without overwriting earlier outputs:

```sh
HF_HUB_OFFLINE=1 uv run erag evaluate-answers --output artifacts/my-answer-check.json
uv run pytest -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

The initial and revised-prompt comparisons are under `artifacts/answer-evaluation-v1` and
`artifacts/answer-evaluation-v2`; earlier failed runs are retained.
Both models receive identical retrieved evidence; comparison latency measures generation only.
Both models passed the revised 15 factual and five abstention checks; generation medians
were 9.67 s (14B) and 6.11 s (8B). Archived evidence: `docs/results/answers-phase-1.1`.
The checker uses expected-fact patterns, exact quote provenance and explicit abstention checks;
its engineering smoke thresholds are separate from the historical retrieval targets.
See [architecture](docs/architecture.md) and [current progress](docs/progress.md).

---

Historical retrieval status: **the stricter Stage 1.1 quality gate failed**. No failed candidate was
promoted. Final candidate configuration: `artifacts/revision-v7/selected.yaml`; run its quality
test with `ERAG_QUALITY_CONFIG` and `ERAG_QUALITY_EXPERIMENT=artifacts/revision-v7`.
See `docs/progress.md` for the pending decision and exact replay command.

# Historical Stage 1.1 retrieval quality checks

The historical **chunk-level** targets remain configured in `config/quality.yaml`:
Hit@1 >= 95%, Hit@3 >= 98%, Recall@5 >= 95%, MRR >= .95, nDCG@10 >= .95.
All five must pass. The old stage report and archived gates used earlier weaker targets;
they are historical, not evidence that the current targets pass.

Run the real ranking regression alongside unit/integration tests:

```sh
HF_HUB_OFFLINE=1 ERAG_MODEL_TESTS=1 ERAG_QUALITY_TESTS=1 uv run pytest -q
```

The quality test prints the five-row target/result/pass table at the end and fails the suite
if a target is missed. It requires a built index matching the frozen profile and cached models.
It checks source/chunk hashes, target values, frozen configuration and validation-label hashes
before retrieving. Set `ERAG_QUALITY_CONFIG` and `ERAG_QUALITY_EXPERIMENT` to test a frozen
candidate profile; current candidate paths are in `docs/progress.md`. To include Docker server
parity, also set `ERAG_QDRANT_URL=http://127.0.0.1:16333` after starting the pinned server.

For a CLI evaluation that prints the table and returns a failing exit status:

```sh
uv run erag evaluate --split heldout-v4 --experiment-id quality-replay-001 --output artifacts/quality-replay-001.json --check-targets
```

This replays observed evaluation questions; it is not another independent validation. The
new-phrasing validation set shares source facts/intents with development data, as documented
in its manifest. Five no-answer cases are excluded from relevance metric means.

Candidate improvements retain complete short heading sections and support configurable
BM25/vector RRF weights plus optional sentence-window reranking. The approved BGE reranker is
pinned by commit. Failed attempts and original labels are retained; no query-specific retrieval
rules were introduced during those retrieval experiments. Default configuration is promoted only after passing.

---

# Enterprise RAG: Phase 1 / Stage 1.1

A Python CLI for retrieval and ranking over exactly five synthetic DevOps documents. Supports
BM25, Qdrant cosine vector search, hybrid reciprocal rank fusion, and optional cross-encoder
reranking. The answer extension is described above. See the historical [stage report](docs/stage-reports/stage-1.1.md),
[progress](docs/progress.md), [architecture](docs/architecture.md), and
[evaluation method](docs/evaluation-methodology.md).

Run commands from this repository root with Python 3.13 and uv:

```sh
uv sync --locked
uv run erag index
uv run erag validate
uv run erag search 'What is the IP address of pe-ledger-prod-01?' --mode bm25
uv run erag search 'Why are running pods returning HTTP 503?' --mode hybrid --rerank --top-k 5
```

Initial vector indexing downloads the pinned embedding model. Reranking downloads its pinned
model on first use. Models run on CPU; no API keys, paid services or model training required.
After downloads, use `HF_HUB_OFFLINE=1` for offline inference. BM25 search does not load models.
Default CLI mode remains `hybrid`; the tuning-selected benchmark strategy is reported separately.
All search output is JSON with source text, offsets, metadata, component scores, initial/final
ranks and timings. Structured event logs go to stderr. Scores from different models are not
probabilities or interchangeable scales.

Configuration: [config/default.yaml](config/default.yaml). Models are pinned by commit and
transitive packages by `uv.lock`; use `--locked` to reproduce that environment. Rebuild after
changing sources, chunking parameters or embedding identity. The tool rejects stale indexes.
A build publishes one canonical chunk snapshot only after Qdrant payload verification succeeds.

Updates: edit one of the five existing `.txt` files, then run `uv run erag index`. Source ID stays
stable, content version changes, and outdated chunks disappear from both active indexes.

```sh
uv run erag delete certificate-management
uv run erag validate
uv run erag restore
```

Deletion is logical: it removes a document from the active collection, persists a tombstone
across rebuilds, and keeps the five source fixtures on disk. `restore` restores all five.
Old Qdrant collections are retained for consistent readers; this is not physical data erasure.
Do not evaluate a deleted collection against the complete five-document relevance labels.

Embedded Qdrant stores data under `artifacts/index/qdrant` and permits one process per storage
directory. Run CLI commands sequentially. Server integration is verified with the pinned
Qdrant image. On a machine with Docker Compose, use:

```sh
docker compose up -d --wait
uv run erag --qdrant-url http://127.0.0.1:16333 index
uv run erag --qdrant-url http://127.0.0.1:16333 search 'private DNS ledger' --mode hybrid
ERAG_QDRANT_URL=http://127.0.0.1:16333 uv run pytest -q tests/test_real_providers.py
docker compose down
```

If this Docker installation lacks Compose, the tested equivalent is:

```sh
docker run --detach --name erag-stage11-qdrant --publish 127.0.0.1:16333:6333 --volume erag-stage11-qdrant-data:/qdrant/storage --env QDRANT__TELEMETRY_DISABLED=true --health-cmd 'bash -c "echo > /dev/tcp/127.0.0.1/6333"' --health-interval 2s --health-timeout 2s --health-retries 30 qdrant/qdrant:v1.20.0@sha256:9bfc89f9914fc14876a88c05270685921a55b8877b254a98bcc1cc2f0a424cd2
HF_HUB_OFFLINE=1 ERAG_MODEL_TESTS=1 ERAG_QDRANT_URL=http://127.0.0.1:16333 uv run pytest -q
HF_HUB_OFFLINE=1 uv run python scripts/verify_server.py
docker stop erag-stage11-qdrant
docker rm erag-stage11-qdrant
```

The Compose command itself was unavailable here. Automatic approval review rejected downloading
a separate Compose executable, so server verification used Docker directly. No system-wide
Compose installation was changed. The supplied Compose file pins the same tested image digest.

The server and embedded backend share a manifest location by default; switching backend
requires rebuilding. Use separate configuration/state paths to keep both active.
Compose binds only local port 16333 and keeps its storage in a named volume.

Tests and static validation:

```sh
uv run pytest -q
HF_HUB_OFFLINE=1 ERAG_MODEL_TESTS=1 uv run pytest -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run python scripts/chunk_audit.py
```

Default tests use deterministic test doubles plus real embedded Qdrant. The opt-in model test
uses the actual cached models and checks ingestion, indexing, all retrieval modes and reranking.
Server integration is opt-in and skips unless `ERAG_QDRANT_URL` is set. Skips are not passes.

Reproduce the preregistered three-run comparison and final held-out evaluation:

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run python scripts/experiment.py
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run python scripts/verify_reproducibility.py
```

Use `--output artifacts/reproduction` if `artifacts/experiments` already exists. Experiment files
are never overwritten. The sweep fixes overlap at 20 words, compares 100/140/180 word budgets,
compares all six retrieval/reranking variants, then freezes selection before opening held-out
labels. The overlap audit is structural only, not another retrieval tuning experiment.
For a single explicit-split run:

```sh
uv run erag evaluate --split tuning --experiment-id manual-001 --output artifacts/manual-001.json
```

Detailed raw experiment reports and derived chunk labels are preserved under `docs/results`.
Sources and labels are versioned under `data/stage-1.1`; labels must not be edited to improve
scores. Small synthetic data and hand-authored evidence labels limit these results. Stage 1.2
and production readiness are not implied by Stage 1.1 metrics.
