> Historical result under the original thresholds. The user subsequently required stricter
> five-metric chunk targets; current status is tracked in `docs/progress.md` and the new
> machine-readable validation results. This historical report is not being regenerated.

# Stage 1.1 report — 2026-10-24

**Stage:** 1.1. **Documents:** exactly 5, dataset `stage-1.1-v1`.
**Implementation:** Completed. All retrieval modules and CLI implemented.
**Automated tests:** 53 passed, 0 skipped, 0 failed, 0 errors.
Real-model integration and Docker Qdrant server parity checks executed successfully; no tests skipped.
**Retrieval quality gate:** Passed for the tuning-selected `bm25` strategy.
**Overall quality gate:** Passed; see [recorded checks](../results/stage-quality-gate.json).
**Recommendation:** Ready for Stage 1.2, subject to explicit approval.
No production-readiness claim.

## Protocol and selected configuration

User approved the thresholds and three tuning iterations. Compared 100/140/180 word limits,
fixed 20-word overlap, 20 candidates, RRF k=60, CPU inference with two threads. Six strategies
were evaluated per configuration. Selected **BM25, 140 words, overlap 20, 29 chunks** using tuning
results only. Selection ordered: passing gates, chunk nDCG@10, document nDCG@5, chunk Recall@5,
lower p95 latency, then smaller chunks. `frozen-selection.json` was written before held-out labels
were opened. The CLI default remains hybrid as requested. No post-held-out tuning was performed.

18 tuning and 18 held-out queries each contain 15 answerable and 3 no-answer cases. Means below
use the 15 answerable cases. Labels were generated from literal facts and validated against
sources; independent human review remains advisable. Queries authored alongside the corpus
are a separate split, not a blinded external benchmark. Small sample size limits confidence.

## Held-out results

| Strategy | Doc R@3 | Doc R@5 | Doc MRR | Doc nDCG@5 | Chunk R@3 | Chunk R@5 | Chunk R@10 | Chunk MRR | Chunk nDCG@5 | Chunk nDCG@10 | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 1.000 | 1.000 | 1.000 | 0.975 | 0.956 | 1.000 | 1.000 | 0.933 | 0.923 | 0.923 | 0.18 |
| bm25+rerank | 1.000 | 1.000 | 1.000 | 0.967 | 0.967 | 1.000 | 1.000 | 0.967 | 0.933 | 0.933 | 177.51 |
| vector | 0.933 | 1.000 | 0.967 | 0.938 | 0.933 | 0.933 | 0.967 | 0.933 | 0.879 | 0.895 | 4.65 |
| vector+rerank | 1.000 | 1.000 | 1.000 | 0.964 | 0.933 | 0.967 | 0.967 | 0.967 | 0.919 | 0.919 | 175.78 |
| hybrid | 0.967 | 1.000 | 1.000 | 0.972 | 0.967 | 0.967 | 0.967 | 1.000 | 0.964 | 0.964 | 4.12 |
| hybrid+rerank | 1.000 | 1.000 | 1.000 | 0.967 | 0.967 | 1.000 | 1.000 | 0.967 | 0.933 | 0.933 | 175.66 |

All reported values are measured with the same frozen chunk configuration, not selected
individually per strategy. No-answer queries have undefined relevance metrics, so they are
excluded from these means. The tool returns ranked candidates for them and makes no
abstention claim. Full no-answer rankings and scores are retained in the raw report.

Agreed thresholds: document R@3 >= .90, MRR >= .85, nDCG@5 >= .85; chunk R@5 >= .85,
nDCG@10 >= .80; warm p95 <= 2000 ms without reranking or <= 5000 ms with reranking.
Selected BM25 satisfies every numerical gate. Document R@5 is a weak metric with only five documents.

## Individual retrieval misses and diagnosis

A miss here means any relevant document missing from top 3 or any evidence-labelled chunk
missing from top 5, stricter than the aggregate gate. Selected BM25 had **no** such held-out
misses; this does not mean perfect graded ranking. Its document nDCG@5 is below 1.

- **heldout-05 / vector:** What must I verify before increasing AKS nodes during upgrades? Expected: `c3660bde-fbe9-5985-8421-ca08e96d4a03` grade 3, rank 1, `d51466c8-46d2-5691-80be-c7251445c402` grade 2, rank absent.
- **heldout-05 / vector+rerank:** What must I verify before increasing AKS nodes during upgrades? Expected: `c3660bde-fbe9-5985-8421-ca08e96d4a03` grade 3, rank 1, `d51466c8-46d2-5691-80be-c7251445c402` grade 2, rank absent.
- **heldout-05 / hybrid:** What must I verify before increasing AKS nodes during upgrades? Expected: `c3660bde-fbe9-5985-8421-ca08e96d4a03` grade 3, rank 1, `d51466c8-46d2-5691-80be-c7251445c402` grade 2, rank 16.
- **heldout-15 / vector:** Does opening HTTPS repair a certificate hostname mismatch? Expected: `90310603-4d83-5736-aece-f47d0b6aedeb` grade 3, rank 9, `d883c51e-dcd3-5eba-b1b0-c0826fe4333f` grade 2, rank 1.

For heldout-05, vector retrieval misses the secondary networking passage at candidate depth 20;
reranking cannot recover an absent candidate. Hybrid includes that passage at rank 16 and
reranking brings it into the top five. For heldout-15, vector retrieval ranks the directly
relevant networking passage ninth behind several related TLS passages; reranking improves it.
Labels and sources were not changed in response. These are diagnosed limitations, not reasons
to tune on held-out data. Future investigation requires a fresh held-out set and approved scope.

## Chunking validation

| Max words | Overlap | Chunks | Missing words | Extra overlap occurrences | Preserved blocks |
|---:|---:|---:|---:|---:|---:|
| 100 | 0 | 30 | 0 | 0 | 70/70 |
| 100 | 20 | 30 | 0 | 500 | 70/70 |
| 140 | 0 | 19 | 0 | 0 | 70/70 |
| 140 | 20 | 29 | 0 | 480 | 70/70 |
| 140 | 40 | 29 | 0 | 960 | 70/70 |
| 180 | 20 | 15 | 0 | 200 | 70/70 |

Zero missing non-whitespace source content, exact duplicate chunks, or invalid offsets/metadata
in tested configurations. All 35 headings and 70 paragraph/heading/metadata blocks are preserved
in at least one chunk. Intentional overlap is counted separately from duplicate exact chunks.
The audit initially omitted the last block in each file due to trailing newlines; corrected
before final reporting. Word-level coverage validation always included the full files.
Overlap-only variants are integrity audits, not additional retrieval tuning runs. No claim that
20-word overlap is the globally optimal overlap. Heading metadata names the active heading at
the start of each chunk; a chunk may also contain subsequent headings preserved in its text.

## Performance and reproducibility

Platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`, `arm64`, Python 3.13.13, two CPU inference threads.
CPU model name unavailable in the sandbox. These timings are local measurements, not a
server or enterprise-load benchmark. p95 uses 18 warm queries per strategy; no throughput claim.
Model load: 3361.5 ms.
Selected index build: 247.4 ms for 5 documents / 29 chunks,
excluding model load and Qdrant client initialization.
The regression replay reproduced all 108 frozen tuning rankings and scores within 1e-5.

Embedding: `sentence-transformers/multi-qa-MiniLM-L6-cos-v1@b207367332321f8e44f96e224ef15bc607f4dbf0`.
Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2@233902d25c440f23af6f7d6e94d2946bac0bee0a`.
Packages: qdrant-client=1.20.0, rank-bm25=0.2.2, sentence-transformers=6.2.0, torch=2.14.1, transformers=5.20.2, numpy=2.5.4, PyYAML=6.0.3, pytest=9.1.1.
Lock SHA-256: `6d874e1d1e532f9242f4a3aaba4e4451863dc6e16ec98d3d326cbb446246a187`. Corpus SHA-256: `82a195647c98ce0e155023fce27984de2cabd1f62265c0076547fb32a525467e`.
Source hashes are recorded per report. The repository had no initial commit; the original
report's git_revision value `HEAD` is not a real revision. This diagnostic was corrected later.
After evaluation only validation hardening, error handling, deprecated-property access and
embedded-search warning handling changed; no scoring, candidate, label or selection parameter changed. Regression replay
and tests validate those maintenance changes without reusing held-out data for selection.

## Server validation and remaining limitations

Official server image: `qdrant/qdrant:v1.20.0@sha256:9bfc89f9914fc14876a88c05270685921a55b8877b254a98bcc1cc2f0a424cd2`.
Real server version: `1.20.0`. User explicitly approved
image download and local execution. The image is pinned by digest in `compose.yaml`.
The complete suite passed against the server. An additional real-model smoke check verified
all six strategy/reranking variants against embedded-backend rankings, deterministic rebuilds,
updates, logical deletion, and restore. Server index creation took 445.5 ms
in this smoke check, not a load benchmark. Both backends used the same corpus and chunk hashes.
No held-out labels were used for these checks.

The extra smoke paraphrase "What IP address does pe-ledger-prod-01 have?" retrieves the expected
networking source second with unreranked BM25 on both backends; reranking moves it first.
An initial smoke assertion assumed first place and failed. Its original log is preserved;
the server parity check uses identical full rankings and the agreed top-three document cutoff.
Official evaluation labels, thresholds, selected configuration and held-out results are unchanged.

- This machine lacks Docker Compose. Automatic approval review rejected downloading its CLI as
  an additional dependency. The server was successfully tested using the existing Docker CLI
  with the image, loopback port, volume, environment and health-check settings in `compose.yaml`.
  The Compose file is supplied, but the Compose invocation itself has not executed here.
- Embedded Qdrant permits one process per storage directory; the server path is needed for
  concurrent processes. CLI reports the lock error instead of hiding it.
- Deletion affects both active indexes and persists a tombstone; source files and old immutable
  Qdrant collections remain. Physical erasure and collection garbage collection are not implemented.
- Stage 1.1 intentionally rejects any sixth source and rejects physical removal of fixture files;
  logical deletion is the supported CLI operation. Updates rebuild a consistent new collection.
- Strict evidence-span labels are auditable but not exhaustive judgements of every potentially
  relevant passage. This may penalise useful alternative passages.
- Synthetic five-document results, small held-out sample and exact Qdrant search do not establish
  large-corpus performance. No no-answer rejection threshold has been calibrated.

## Reproduce

Run from the repository root:

```sh
uv sync --locked
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
HF_HUB_OFFLINE=1 ERAG_MODEL_TESTS=1 uv run pytest -q
uv run python scripts/chunk_audit.py
HF_HUB_OFFLINE=1 uv run erag index
uv run erag validate
uv run erag search '10.42.8.17' --mode bm25 --top-k 3
HF_HUB_OFFLINE=1 uv run erag search 'running pods HTTP 503' --mode hybrid --rerank
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run python scripts/experiment.py --output artifacts/reproduction
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run python scripts/verify_reproducibility.py
```

The replay command defaults to the original `artifacts/experiments` directory; pass
`--experiments artifacts/reproduction` to replay a reproduction. Remove HF_HUB_OFFLINE for
first downloads. For complete server integration on a machine with Compose:

```sh
docker compose up -d --wait
HF_HUB_OFFLINE=1 ERAG_MODEL_TESTS=1 ERAG_QDRANT_URL=http://127.0.0.1:16333 uv run pytest -q
HF_HUB_OFFLINE=1 uv run python scripts/verify_server.py
docker compose down
```

Without Compose, use the equivalent exact `docker run` command documented in README.
The verified test container was stopped and removed after the checks; its pinned image remains cached.

Raw reports: [results directory](../results/README.md). Compact per-query metrics and exact
labelled chunk IDs: [evaluation-summary.json](../results/evaluation-summary.json).
