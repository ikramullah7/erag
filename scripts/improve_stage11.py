"""One development-only fusion sweep, then frozen fresh validation; prints five targets."""

import gzip
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from erag.config import load_config
from erag.diagnostics import configure_logging
from erag.evaluation import (
    evaluate,
    format_target_table,
    load_quality_targets,
    quality_gate,
    ranking_metrics,
)
from erag.fusion import rrf
from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.pipeline import Engine, atomic_json
from erag.storage import QdrantStore


def main() -> int:
    root = Path("artifacts/revision-v2")
    if root.exists():
        raise ValueError("revision-v2 already exists; preserve its frozen validation")
    root.mkdir()
    configure_logging()
    manifest = json.loads(Path("data/stage-1.1/evaluation-v2-manifest.json").read_text())
    # Verify frozen input bytes before the development sweep. Do not read fresh query contents.
    for name, expected in manifest["label_hashes"].items():
        assert hashlib.sha256((Path("data/stage-1.1") / name).read_bytes()).hexdigest() == expected
    development = []
    for name in ("tuning-2", "heldout"):
        archive = Path("docs/results") / f"{name}.json.gz"
        original = json.loads(gzip.decompress(archive.read_bytes()))
        development.extend(
            row
            for row in original["queries"]
            if row["mode"] == "hybrid" and not row["reranked"] and row["answerable"]
        )
    targets = load_quality_targets()
    trials = []
    # Only lexical RRF weight changes. Corpus, chunking, models, rank constant and depth fixed.
    for weight in (0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4):
        per_query = []
        for row in development:
            components = row["diagnostics"]["components"]
            fused = rrf([components["bm25"], components["cosine"]], 60, [weight, 1])[:20]
            metrics = ranking_metrics([cid for cid, _ in fused], row["relevant_chunks"], [5, 10])
            per_query.append({"id": row["case_id"], "metrics": metrics})
        means = {
            metric: sum(row["metrics"][metric] for row in per_query) / len(per_query)
            for metric in targets
        }
        passes = all(means[metric] >= threshold for metric, threshold in targets.items())
        trials.append(
            {
                "bm25_weight": weight,
                "vector_weight": 1,
                "means": means,
                "targets_passed": passes,
                "per_query": per_query,
            }
        )
    selected = max(
        trials,
        key=lambda t: (
            t["targets_passed"],
            t["means"]["ndcg@10"],
            t["means"]["hit@1"],
            t["means"]["mrr"],
            -t["bm25_weight"],
        ),
    )
    atomic_json(root / "development-sweep.json", {"iteration": 1, "trials": trials})
    cfg = replace(
        load_config("config/default.yaml"), bm25_weight=selected["bm25_weight"], vector_weight=1
    )
    atomic_json(
        root / "frozen-selection.json",
        {
            "config": asdict(cfg),
            "mode": "hybrid",
            "rerank": False,
            "development_cases": len(development),
            "original_heldout_retired": True,
            "fresh_label_hash": manifest["label_hashes"]["heldout-v2.json"],
            "selection_rule": "all five development targets, then nDCG10, Hit1, MRR, lower weight",
        },
    )
    (root / "selected.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    store = QdrantStore(path=str(Path(cfg.state) / "qdrant"))
    try:
        engine = Engine(cfg, store, SentenceEmbedder(cfg), CrossEncoderReranker(cfg))
        dev = evaluate(
            engine,
            Path(cfg.dataset) / "tuning-v2.json",
            root / "development.json",
            "stage-1.1-v2-development",
            "tuning-v2",
        )
        # Both simulations and actual retrieval must reproduce the selected development metrics.
        actual = dev["summary"]["hybrid"]
        for metric in targets:
            assert abs(actual["chunk"][metric] - selected["means"][metric]) < 1e-10
        if not quality_gate(actual, False)["passed"]:
            print("Development gates failed; fresh validation was not opened.")
            print(format_target_table(actual))
            return 2
        heldout = evaluate(
            engine,
            Path(cfg.dataset) / "heldout-v2.json",
            root / "validation.json",
            "stage-1.1-v2-fresh-validation",
            "heldout-v2",
        )
        summary = heldout["summary"]["hybrid"]
        gate = quality_gate(summary, False)
        atomic_json(root / "quality-gate.json", {"strategy": "hybrid", "targets": targets, **gate})
        print("Stage 1.1 | fresh validation | hybrid | chunk level | 30 answerable queries")
        print(format_target_table(summary))
        print(f"Quality gate: {'PASS' if gate['passed'] else 'FAIL'}")
        return 0 if gate["passed"] else 2
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
