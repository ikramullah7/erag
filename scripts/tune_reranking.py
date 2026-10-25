"""Third improvement iteration: evidence-window reranking on development only."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from erag.config import load_config
from erag.diagnostics import configure_logging
from erag.evaluation import evaluate, load_quality_targets, quality_gate
from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.pipeline import Engine, atomic_json
from erag.storage import QdrantStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--output", default="artifacts/revision-v4")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise ValueError("Do not overwrite tuning measurements")
    root.mkdir()
    configure_logging()
    cfg = replace(
        load_config("artifacts/revision-v3/config.yaml"), rerank_window_sentences=args.sentences
    )
    (root / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    store = QdrantStore(path=str(Path(cfg.state) / "qdrant"))
    try:
        engine = Engine(cfg, store, SentenceEmbedder(cfg), CrossEncoderReranker(cfg))
        report = evaluate(
            engine,
            Path(cfg.dataset) / "tuning-v3.json",
            root / "development.json",
            "stage-1.1-v4-window-development",
            "tuning-v3",
        )
        targets = load_quality_targets()
        summary = {
            name: {
                "chunk": {k: s["chunk"][k] for k in targets},
                "gate": quality_gate(s, name.endswith("+rerank")),
            }
            for name, s in report["summary"].items()
        }
        atomic_json(root / "development-summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
