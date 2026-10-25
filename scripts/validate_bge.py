"""Freeze the development-qualified profile, then open validation v3 once."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from erag.config import load_config
from erag.diagnostics import configure_logging
from erag.evaluation import evaluate, format_target_table, load_quality_targets, quality_gate
from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.pipeline import Engine, atomic_json
from erag.storage import QdrantStore


def main() -> int:
    root = Path("artifacts/revision-v5")
    if (root / "validation.json").exists() or (root / "frozen-selection.json").exists():
        raise ValueError("Do not overwrite or reopen frozen validation")
    configure_logging()
    dev = json.loads((root / "development.json").read_text())
    assert quality_gate(dev["summary"]["hybrid+rerank"], True)["passed"]
    cfg = replace(load_config(root / "config.yaml"), rerank=True)
    manifest = json.loads(Path("data/stage-1.1/evaluation-v3-manifest.json").read_text())
    cases = Path(cfg.dataset) / "heldout-v3.json"
    expected = manifest["label_hashes"]["heldout-v3.json"]
    assert hashlib.sha256(cases.read_bytes()).hexdigest() == expected
    atomic_json(
        root / "frozen-selection.json",
        {
            "config": asdict(cfg),
            "strategy": "hybrid+rerank",
            "targets": load_quality_targets(),
            "validation_hash": expected,
            "selection_reason": "all five development targets passed before validation access",
        },
    )
    (root / "selected.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    store = QdrantStore(path=str(Path(cfg.state) / "qdrant"))
    try:
        engine = Engine(cfg, store, SentenceEmbedder(cfg), CrossEncoderReranker(cfg))
        result = evaluate(
            engine,
            cases,
            root / "validation.json",
            "stage-1.1-v5-independent-validation",
            "heldout-v3",
            strategies=(("hybrid", True),),
        )
        summary = result["summary"]["hybrid+rerank"]
        gate = quality_gate(summary, True)
        atomic_json(
            root / "quality-gate.json",
            {
                "strategy": "hybrid+rerank",
                "split": "heldout-v3",
                "targets": load_quality_targets(),
                **gate,
            },
        )
        print("Stage 1.1 | independent validation | 30 answerable queries | chunk level")
        print(format_target_table(summary))
        print(f"Quality gate: {'PASS' if gate['passed'] else 'FAIL'}")
        return 0 if gate["passed"] else 2
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
