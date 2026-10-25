"""Three tuning runs, frozen selection, then held-out once. No held-out tuning."""

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from erag.config import load_config
from erag.diagnostics import configure_logging
from erag.evaluation import evaluate, quality_gate
from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.pipeline import Engine, atomic_json, build_index
from erag.storage import QdrantStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/experiments")
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError("output exists; use a new directory to preserve experiment records")
    out.mkdir(parents=True)
    configure_logging()
    cfg = load_config("config/default.yaml")
    started = time.perf_counter()
    embedder, reranker = SentenceEmbedder(cfg), CrossEncoderReranker(cfg)
    atomic_json(out / "model-load.json", {"ms": (time.perf_counter() - started) * 1000})
    options = []
    # Fixed, preregistered sweep: only max_words changes.
    for i, size in enumerate((100, 140, 180), 1):
        run_cfg = replace(cfg, max_words=size, state=str(out / f"index-{i}"))
        store = QdrantStore(path=str(Path(run_cfg.state) / "qdrant"), url=run_cfg.qdrant_url)
        try:
            build_index(run_cfg, embedder, store)
            engine = Engine(run_cfg, store, embedder, reranker)
            report = evaluate(
                engine,
                Path(cfg.dataset) / "tuning.json",
                out / f"tuning-{i}.json",
                f"stage-1.1-tuning-{i}",
                "tuning",
            )
            for strategy, summary in report["summary"].items():
                # Predefined selection: satisfy gates first, then graded chunk ranking,
                # document ranking, recall, lower latency and smaller chunks as tie-breakers.
                key = (
                    quality_gate(summary, strategy.endswith("+rerank"))["passed"],
                    summary["chunk"]["ndcg@10"],
                    summary["document"]["ndcg@5"],
                    summary["chunk"]["recall@5"],
                    -summary["latency_ms_p95"],
                    -size,
                )
                options.append((key, run_cfg, strategy))
        finally:
            store.close()
    _, selected, strategy = max(options, key=lambda row: row[0])
    frozen = {
        "config": asdict(selected),
        "strategy": strategy,
        "selection_rule": "gates, chunk ndcg10, doc ndcg5, chunk recall5, latency, size",
    }
    atomic_json(out / "frozen-selection.json", frozen)
    (out / "selected.yaml").write_text(yaml.safe_dump(asdict(selected), sort_keys=False))
    # The held-out file is first opened only after publishing frozen-selection.json.
    store = QdrantStore(path=str(Path(selected.state) / "qdrant"), url=selected.qdrant_url)
    try:
        report = evaluate(
            Engine(selected, store, embedder, reranker),
            Path(cfg.dataset) / "heldout.json",
            out / "heldout.json",
            "stage-1.1-heldout",
            "heldout",
        )
        gate = quality_gate(report["summary"][strategy], strategy.endswith("+rerank"))
        atomic_json(out / "quality-gate.json", {"strategy": strategy, **gate})
        print(
            json.dumps(
                {"strategy": strategy, "max_words": selected.max_words, "heldout_gate": gate},
                indent=2,
            )
        )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
