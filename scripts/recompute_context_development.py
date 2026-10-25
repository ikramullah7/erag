"""Recompute full-context scores only from the explicitly designated development file."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from erag.config import load_config
from erag.diagnostics import configure_logging
from erag.evaluation import evaluate
from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.pipeline import Engine
from erag.storage import QdrantStore


def main() -> None:
    root = Path("artifacts/revision-v7")
    root.mkdir(exist_ok=True)
    if (root / "context-development.json").exists():
        raise ValueError("Preserve development measurements")
    cases = Path("data/stage-1.1/tuning-v4.json")
    manifest = json.loads(Path("data/stage-1.1/evaluation-v4-manifest.json").read_text())
    assert (
        hashlib.sha256(cases.read_bytes()).hexdigest() == manifest["label_hashes"]["tuning-v4.json"]
    )
    development = json.loads(cases.read_text())
    assert all(not row["id"].startswith("heldout-v4-") for row in development)
    configure_logging()
    cfg = replace(
        load_config("artifacts/revision-v6/selected.yaml"),
        rerank_window_sentences=0,
        rerank_fusion=False,
    )
    store = QdrantStore(path=str(Path(cfg.state) / "qdrant"))
    try:
        engine = Engine(cfg, store, SentenceEmbedder(cfg), CrossEncoderReranker(cfg))
        evaluate(
            engine,
            cases,
            root / "context-development.json",
            "stage-1.1-approved-final-development",
            "tuning-v4",
            strategies=(("hybrid", True),),
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
