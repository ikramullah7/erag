"""Final approved BGE round: late fusion uses development measurements only."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from erag.config import load_config
from erag.evaluation import load_quality_targets, ranking_metrics
from erag.fusion import rrf
from erag.pipeline import atomic_json


def main() -> None:
    root = Path("artifacts/revision-v7")
    if (root / "frozen-selection.json").exists():
        raise ValueError("Selection already frozen")
    development_path = Path("data/stage-1.1/tuning-v4.json")
    development_hash = hashlib.sha256(development_path.read_bytes()).hexdigest()
    allowed = {row["id"] for row in json.loads(development_path.read_text()) if row["evidence"]}
    assert len(allowed) == 90 and not any(key.startswith("heldout-v4-") for key in allowed)
    context_report = json.loads((root / "context-development.json").read_text())
    window_report = json.loads(Path("artifacts/revision-v6/development.json").read_text())
    assert context_report["split"] == window_report["split"] == "tuning-v4"
    assert context_report["labels_hash"] == window_report["labels_hash"] == development_hash
    context = {row["case_id"]: row for row in context_report["queries"] if row["answerable"]}
    windows = {row["case_id"]: row for row in window_report["queries"] if row["answerable"]}
    assert set(context) == set(windows) == allowed
    targets = load_quality_targets()
    trials = []
    for weight in (0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 4):
        queries = []
        for key in sorted(allowed):
            row = windows[key]
            rankings = [
                context[key]["diagnostics"]["final_ranking"],
                row["diagnostics"]["final_ranking"],
            ]
            weights = [1, 1]
            if weight:
                rankings.append(row["diagnostics"]["initial_ranking"])
                weights.append(weight)
            fused = rrf(rankings, 60, weights)
            metrics = ranking_metrics([cid for cid, _ in fused], row["relevant_chunks"], [5, 10])
            queries.append({"id": key, "metrics": metrics})
        means = {
            metric: sum(row["metrics"][metric] for row in queries) / len(queries)
            for metric in targets
        }
        trials.append(
            {
                "initial_weight": weight,
                "means": means,
                "passed": all(means[k] >= v for k, v in targets.items()),
                "queries": queries,
            }
        )
    best = max(
        trials,
        key=lambda t: (
            t["passed"],
            t["means"]["ndcg@10"],
            t["means"]["hit@1"],
            t["means"]["mrr"],
            -t["initial_weight"],
        ),
    )
    assert best["passed"]
    atomic_json(
        root / "development-sweep.json",
        {
            "approved_bge_iteration": 3,
            "development_hash": development_hash,
            "development_cases": 90,
            "validation_ids_excluded": True,
            "trials": trials,
        },
    )
    cfg = replace(
        load_config("artifacts/revision-v6/selected.yaml"),
        rerank_fusion=True,
        rerank_initial_weight=best["initial_weight"],
    )
    (root / "selected.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    manifest = json.loads(Path("data/stage-1.1/evaluation-v4-manifest.json").read_text())
    validation = "heldout-v4.json"
    atomic_json(
        root / "frozen-selection.json",
        {
            "config": asdict(cfg),
            "strategy": "hybrid+rerank",
            "targets": targets,
            "validation_file": validation,
            "validation_hash": manifest["label_hashes"][validation],
            "corpus_hash": window_report["corpus_hash"],
            "chunk_hash": window_report["chunk_hash"],
            "validation_already_observed": True,
            "selection_inputs": "Two explicitly labelled development-only measurements; 90 whitelisted development IDs. Validation-result files are not loaded.",
            "selection_reason": "All five development targets, then nDCG10, Hit1, MRR, lower initial weight",
            "validation_protocol": "Observed-question regression replay; not independent validation",
        },
    )
    print(
        json.dumps(
            {"initial_weight": best["initial_weight"], "development": best["means"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
