"""Regression-only replay of frozen tuning results, with no selection or label changes."""

import json
from pathlib import Path

import numpy as np

from erag.config import load_config
from erag.evaluation import environment
from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.pipeline import Engine, atomic_json
from erag.storage import QdrantStore


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="artifacts/experiments")
    args = parser.parse_args()
    root = Path(args.experiments)
    cfg = load_config(root / "selected.yaml")
    reports = [json.loads(p.read_text()) for p in sorted(root.glob("tuning-*.json"))]
    report = next(r for r in reports if r["config"]["max_words"] == cfg.max_words)
    store = QdrantStore(path=str(Path(cfg.state) / "qdrant"))
    checked = 0
    try:
        engine = Engine(cfg, store, SentenceEmbedder(cfg), CrossEncoderReranker(cfg))
        for row in report["queries"]:
            original = row["diagnostics"]
            actual = engine.search(original["query"], row["mode"], row["reranked"], cfg.candidates)
            assert [x[0] for x in original["final_ranking"]] == [
                x[0] for x in actual["final_ranking"]
            ]
            assert np.allclose(
                [x[1] for x in original["final_ranking"]],
                [x[1] for x in actual["final_ranking"]],
                atol=1e-5,
            )
            checked += 1
        atomic_json(
            Path("artifacts/reproducibility.json"),
            {
                "passed": True,
                "replayed_rankings": checked,
                "split": "tuning",
                "config_frozen": True,
                "ranking_parameters_changed": False,
                "environment": environment(),
            },
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
