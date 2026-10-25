"""Second improvement iteration: change section boundaries only; no fresh validation access."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from erag.config import load_config
from erag.diagnostics import configure_logging
from erag.evaluation import evaluate, load_quality_targets, quality_gate
from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.pipeline import Engine, atomic_json, build_index
from erag.storage import QdrantStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/revision-v3")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise ValueError("Do not overwrite development iteration")
    root.mkdir()
    configure_logging()
    previous = load_config("artifacts/revision-v2/selected.yaml")
    cfg = replace(previous, section_boundaries=True, state=str(root / "index"))
    (root / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    store = QdrantStore(path=str(Path(cfg.state) / "qdrant"))
    try:
        embedder, reranker = SentenceEmbedder(cfg), CrossEncoderReranker(cfg)
        build_index(cfg, embedder, store)
        report = evaluate(
            Engine(cfg, store, embedder, reranker),
            Path(cfg.dataset) / "tuning-v3.json",
            root / "development.json",
            "stage-1.1-v3-section-development",
            "tuning-v3",
        )
        targets = load_quality_targets()
        compact = {
            name: {
                "chunk": {k: summary["chunk"][k] for k in targets},
                "gate": quality_gate(summary, name.endswith("+rerank")),
            }
            for name, summary in report["summary"].items()
        }
        atomic_json(root / "development-summary.json", compact)
        print(json.dumps(compact, indent=2))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
