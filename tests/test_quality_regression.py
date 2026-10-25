"""Opt-in real quality gate: fail on any of the user's five thresholds."""

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pytest
from conftest import QUALITY_SUMMARY_KEY

from erag.config import Config, load_config
from erag.evaluation import evaluate, load_quality_targets, quality_gate
from erag.models import CrossEncoderReranker, SentenceEmbedder
from erag.pipeline import Engine, atomic_json
from erag.storage import QdrantStore


@pytest.mark.skipif(
    os.getenv("ERAG_QUALITY_TESTS") != "1",
    reason="set ERAG_QUALITY_TESTS=1 for real ranking quality gate",
)
def test_stage11_meets_all_five_chunk_targets(request):
    root = Path(os.getenv("ERAG_QUALITY_EXPERIMENT", "artifacts/revision-v6"))
    frozen = json.loads((root / "frozen-selection.json").read_text())
    cfg = load_config(os.getenv("ERAG_QUALITY_CONFIG", "config/default.yaml"))
    ignored = {"state", "dataset", "qdrant_url"}
    assert {k: v for k, v in asdict(cfg).items() if k not in ignored} == {
        k: v for k, v in asdict(Config(**frozen["config"])).items() if k not in ignored
    }
    assert frozen["targets"] == load_quality_targets()
    cases = Path(cfg.dataset) / frozen["validation_file"]
    assert hashlib.sha256(cases.read_bytes()).hexdigest() == frozen["validation_hash"]
    primary = root / "validation.json"
    replay = primary.exists()
    observed = bool(frozen.get("validation_already_observed", False))
    output = Path(f"artifacts/quality-replay-{time.time_ns()}.json") if replay else primary
    store = QdrantStore(path=str(Path(cfg.state) / "qdrant"), url=cfg.qdrant_url)
    try:
        engine = Engine(cfg, store, SentenceEmbedder(cfg), CrossEncoderReranker(cfg))
        assert engine.snapshot["corpus_hash"] == frozen["corpus_hash"]
        assert engine.snapshot["chunk_hash"] == frozen["chunk_hash"]
        result = evaluate(
            engine,
            cases,
            output,
            "stage-1.1-quality-replay" if replay else "stage-1.1-frozen-validation",
            cases.stem,
            strategies=((cfg.mode, cfg.rerank),),
        )
        key = cfg.mode + ("+rerank" if cfg.rerank else "")
        summary = result["summary"][key]
        request.config.stash[QUALITY_SUMMARY_KEY] = {
            **summary,
            "_quality_context": "Observed-question regression replay"
            if replay or observed
            else "Frozen new-question validation",
        }
        gate = quality_gate(summary, cfg.rerank)
        gate_path = output.with_suffix(".gate.json") if replay else root / "quality-gate.json"
        atomic_json(
            gate_path,
            {
                "strategy": key,
                "split": cases.stem,
                "targets": load_quality_targets(),
                "validation_replay": replay or observed,
                **gate,
            },
        )
        assert gate["passed"], gate["checks"]
    finally:
        store.close()
