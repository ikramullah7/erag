"""Archive real measurements and generate the Stage 1.1 review report."""

import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from erag.pipeline import atomic_json


def main() -> None:
    root = Path("artifacts/experiments")
    out = Path("docs/results")
    out.mkdir(exist_ok=True)
    reports = [json.loads((root / f"tuning-{i}.json").read_text()) for i in (1, 2, 3)]
    heldout = json.loads((root / "heldout.json").read_text())
    selection = json.loads((root / "frozen-selection.json").read_text())
    gate = json.loads((root / "quality-gate.json").read_text())
    archive = {}
    for p in [*sorted(root.glob("*.json")), Path("artifacts/chunk-audit.json")]:
        raw = p.read_bytes()
        target = out / f"{p.name}.gz"
        target.write_bytes(gzip.compress(raw, mtime=0))
        archive[target.name] = {
            "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
            "compressed_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    atomic_json(out / "archive-manifest.json", archive)
    compact = []
    for report in [*reports, heldout]:
        compact.append(
            {k: v for k, v in report.items() if k != "queries"}
            | {
                "queries": [
                    {k: v for k, v in row.items() if k != "diagnostics"}
                    | {
                        "query": row["diagnostics"]["query"],
                        "timings": row["diagnostics"]["timings"],
                        "final_ranking": row["diagnostics"]["final_ranking"],
                    }
                    for row in report["queries"]
                ]
            }
        )
    atomic_json(out / "evaluation-summary.json", compact)
    for filename in ("chunk-audit-summary.json", "reproducibility.json", "cli-validate.json"):
        (out / filename).write_bytes((Path("artifacts") / filename).read_bytes())
    (out / "tests.xml").write_bytes(Path("artifacts/tests.xml").read_bytes())
    (out / "tests.log").write_bytes(Path("artifacts/tests.log").read_bytes())
    tests = ET.parse(out / "tests.xml").getroot()
    suites = [tests] if tests.tag == "testsuite" else list(tests)
    counts = {
        key: sum(int(s.attrib.get(key, 0)) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    server_path = Path("artifacts/server-verification.json")
    server = json.loads(server_path.read_text()) if server_path.exists() else {}
    replay = json.loads(Path("artifacts/reproducibility.json").read_text())
    audit_details = json.loads(Path("artifacts/chunk-audit.json").read_text())
    stage_checks = {
        "retrieval_and_latency": bool(gate["passed"]),
        "all_tests_executed_and_passed": passed > 0
        and all(counts[k] == 0 for k in ("failures", "errors", "skipped")),
        "server_real_model_and_mutation_checks": bool(server.get("passed")),
        "server_corpus_matches_evaluation": server.get("corpus_hash") == heldout["corpus_hash"]
        and server.get("chunk_hash") == heldout["chunk_hash"],
        "chunk_integrity": all(
            d["missing_words"] == 0
            and d["duplicate_exact_chunks"] == 0
            and d["source_offsets_valid"]
            for a in audit_details
            for d in a["documents"]
        ),
        "reproducible_rankings": bool(replay["passed"]),
    }
    stage_passed = all(stage_checks.values())
    atomic_json(out / "stage-quality-gate.json", {"passed": stage_passed, "checks": stage_checks})
    if server:
        (out / "server-verification.json").write_bytes(server_path.read_bytes())
        (out / "server-initial-smoke.log").write_bytes(
            Path("artifacts/server-initial-smoke.log").read_bytes()
        )
    implementation = "Completed" if stage_passed else "Incomplete: see quality checks"
    readiness = (
        "Ready for Stage 1.2, subject to explicit approval"
        if stage_passed
        else "Further improvements required"
    )
    rows = [
        "| Strategy | Doc R@3 | Doc R@5 | Doc MRR | Doc nDCG@5 | Chunk R@3 | Chunk R@5 | Chunk R@10 | Chunk MRR | Chunk nDCG@5 | Chunk nDCG@10 | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in heldout["summary"].items():
        d, c = s["document"], s["chunk"]
        values = [
            d["recall@3"],
            d["recall@5"],
            d["mrr"],
            d["ndcg@5"],
            c["recall@3"],
            c["recall@5"],
            c["recall@10"],
            c["mrr"],
            c["ndcg@5"],
            c["ndcg@10"],
        ]
        rows.append(
            "| "
            + name
            + " | "
            + " | ".join(f"{v:.3f}" for v in values)
            + f" | {s['latency_ms_p95']:.2f} |"
        )
    failed = []
    for row in heldout["queries"]:
        if row["failed"]:
            ranks = {cid: i for i, (cid, _) in enumerate(row["diagnostics"]["final_ranking"], 1)}
            expected = ", ".join(
                f"`{cid}` grade {g}, rank {ranks.get(cid, 'absent')}"
                for cid, g in row["relevant_chunks"].items()
            )
            failed.append(
                f"- **{row['case_id']} / {row['mode']}{'+rerank' if row['reranked'] else ''}:** "
                f"{row['diagnostics']['query']} Expected: {expected}."
            )
    audit = json.loads(Path("artifacts/chunk-audit-summary.json").read_text())
    auditrows = [
        "| Max words | Overlap | Chunks | Missing words | Extra overlap occurrences | Preserved blocks |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for a in audit:
        auditrows.append(
            f"| {a['max_words']} | {a['overlap_words']} | {a['chunks']} | {a['missing_words']} | "
            f"{a['overlap_word_occurrences']} | {a['preserved_paragraphs']}/{a['total_paragraphs']} |"
        )
    env = heldout["environment"]
    report = f"""# Stage 1.1 report — 2026-10-24

**Stage:** 1.1. **Documents:** exactly 5, dataset `stage-1.1-v1`.
**Implementation:** {implementation}. All retrieval modules and CLI implemented.
**Automated tests:** {passed} passed, {counts["skipped"]} skipped, {counts["failures"]} failed, {counts["errors"]} errors.
Real-model integration and Docker Qdrant server parity checks executed successfully; no tests skipped.
**Retrieval quality gate:** {"Passed" if gate["passed"] else "Failed"} for the tuning-selected `{selection["strategy"]}` strategy.
**Overall quality gate:** {"Passed" if stage_passed else "Failed"}; see [recorded checks](../results/stage-quality-gate.json).
**Recommendation:** {readiness}.
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

{chr(10).join(rows)}

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

{chr(10).join(failed)}

For heldout-05, vector retrieval misses the secondary networking passage at candidate depth 20;
reranking cannot recover an absent candidate. Hybrid includes that passage at rank 16 and
reranking brings it into the top five. For heldout-15, vector retrieval ranks the directly
relevant networking passage ninth behind several related TLS passages; reranking improves it.
Labels and sources were not changed in response. These are diagnosed limitations, not reasons
to tune on held-out data. Future investigation requires a fresh held-out set and approved scope.

## Chunking validation

{chr(10).join(auditrows)}

Zero missing non-whitespace source content, exact duplicate chunks, or invalid offsets/metadata
in tested configurations. All 35 headings and 70 paragraph/heading/metadata blocks are preserved
in at least one chunk. Intentional overlap is counted separately from duplicate exact chunks.
The audit initially omitted the last block in each file due to trailing newlines; corrected
before final reporting. Word-level coverage validation always included the full files.
Overlap-only variants are integrity audits, not additional retrieval tuning runs. No claim that
20-word overlap is the globally optimal overlap. Heading metadata names the active heading at
the start of each chunk; a chunk may also contain subsequent headings preserved in its text.

## Performance and reproducibility

Platform: `{env["platform"]}`, `{env["machine"]}`, Python 3.13.13, two CPU inference threads.
CPU model name unavailable in the sandbox. These timings are local measurements, not a
server or enterprise-load benchmark. p95 uses 18 warm queries per strategy; no throughput claim.
Model load: {json.loads((root / "model-load.json").read_text())["ms"]:.1f} ms.
Selected index build: {heldout["index_ms"]:.1f} ms for 5 documents / {heldout["chunk_count"]} chunks,
excluding model load and Qdrant client initialization.
The regression replay reproduced all 108 frozen tuning rankings and scores within 1e-5.

Embedding: `{heldout["config"]["embedding_model"]}@{heldout["config"]["embedding_revision"]}`.
Reranker: `{heldout["config"]["reranker_model"]}@{heldout["config"]["reranker_revision"]}`.
Packages: {", ".join(f"{k}={v}" for k, v in env["packages"].items())}.
Lock SHA-256: `{env["lock_hash"]}`. Corpus SHA-256: `{heldout["corpus_hash"]}`.
Source hashes are recorded per report. The repository had no initial commit; the original
report's git_revision value `HEAD` is not a real revision. This diagnostic was corrected later.
After evaluation only validation hardening, error handling, deprecated-property access and
embedded-search warning handling changed; no scoring, candidate, label or selection parameter changed. Regression replay
and tests validate those maintenance changes without reusing held-out data for selection.

## Server validation and remaining limitations

Official server image: `{server.get("image", "not verified")}`.
Real server version: `{server.get("server_version", "not verified")}`. User explicitly approved
image download and local execution. The image is pinned by digest in `compose.yaml`.
The complete suite passed against the server. An additional real-model smoke check verified
all six strategy/reranking variants against embedded-backend rankings, deterministic rebuilds,
updates, logical deletion, and restore. Server index creation took {server.get("initial_index_ms", 0):.1f} ms
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
"""
    Path("docs/stage-reports/stage-1.1.md").write_text(report)
    experiment_lines = [
        "# Experiments",
        "",
        "Three approved tuning iterations completed. No fourth tuning iteration.",
        "",
        "| Run | Words/overlap | Chunks | Index ms | Strategy | Doc nDCG@5 | Chunk R@5 | Chunk nDCG@10 | Missed queries |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for i, r in enumerate(reports, 1):
        for strategy, s in r["summary"].items():
            experiment_lines.append(
                f"| tuning-{i} | {r['config']['max_words']}/20 | {r['chunk_count']} | {r['index_ms']:.1f} | "
                f"{strategy} | {s['document']['ndcg@5']:.3f} | {s['chunk']['recall@5']:.3f} | "
                f"{s['chunk']['ndcg@10']:.3f} | {', '.join(s['failed_queries']) or 'none'} |"
            )
    experiment_lines.extend(
        [
            "",
            "140/20 BM25 won the predefined tuning selection rule (chunk nDCG@10 .959).",
            "Increasing word budget to 180 reduced chunk count to 15 but did not improve the selected ranking metric.",
            "Reranking often improved recall but did not uniformly improve graded ranking; tuning-08 regressed at 180 words.",
            "The held-out evaluation was run once after freezing selection. No labels or ranking parameters changed afterward.",
            "",
            "All runs: dataset stage-1.1-v1, exact models/revisions and all config values in reports,",
            "candidate count 20, RRF k=60, overlap 20. Overlap-only comparisons were integrity audits.",
            "See [stage report](stage-reports/stage-1.1.md) and [raw/compact results](results/README.md).",
        ]
    )
    Path("docs/experiments.md").write_text("\n".join(experiment_lines) + "\n")
    (out / "README.md").write_text(
        """# Recorded Stage 1.1 measurements\n\n`evaluation-summary.json` contains configuration, versions, metrics, category summaries, query text,\nexact labelled chunk IDs/grades and final rankings for all four measured runs. Full text and\ncomponent rankings/scores are retained in `tuning-*.json.gz` and `heldout.json.gz`.\n\nRead a raw report using `python3 -m gzip -dc docs/results/heldout.json.gz`.\n`archive-manifest.json` records compressed and original SHA-256 hashes. All archives contain\nactual executions, not synthetic metrics. `tests.log` and `tests.xml` record executed tests and skips.\n\nThe held-out set must not be reused for future tuning. The regression replay uses only tuning queries.\nThe quality-gate archive measures retrieval thresholds; the overall stage gate also requires\nthe operational checks described in the stage report.\n"""
    )


if __name__ == "__main__":
    main()
