"""Second approved BGE iteration: same model, two-sentence evidence windows, development only."""

from dataclasses import asdict, replace
from pathlib import Path

import yaml

from erag.config import load_config
from erag.diagnostics import configure_logging
from erag.evaluation import evaluate, format_target_table, quality_gate
from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.pipeline import Engine, atomic_json
from erag.storage import QdrantStore


def main() -> int:
    root = Path("artifacts/revision-v6")
    if root.exists():
        raise ValueError("Do not overwrite tuning measurements")
    root.mkdir()
    configure_logging()
    cfg = replace(load_config("artifacts/revision-v5/selected.yaml"), rerank_window_sentences=2)
    (root / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    store = QdrantStore(path=str(Path(cfg.state) / "qdrant"))
    try:
        engine = Engine(cfg, store, SentenceEmbedder(cfg), CrossEncoderReranker(cfg))
        report = evaluate(
            engine,
            Path(cfg.dataset) / "tuning-v4.json",
            root / "development.json",
            "stage-1.1-v6-bge-window-development",
            "tuning-v4",
            strategies=(("hybrid", True),),
        )
        summary = report["summary"]["hybrid+rerank"]
        gate = quality_gate(summary, True)
        atomic_json(root / "development-gate.json", gate)
        print("Stage 1.1 | development | BGE evidence windows | 90 answerable queries")
        print(format_target_table(summary))
        return 0 if gate["passed"] else 2
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
