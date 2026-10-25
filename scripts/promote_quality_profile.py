"""Publish a profile only after its real quality test and the complete software suite pass."""

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

import yaml

from erag.config import load_config
from erag.evaluation import format_target_table, load_quality_targets
from erag.models import SentenceEmbedder
from erag.pipeline import atomic_json, build_index
from erag.storage import QdrantStore


def main() -> None:
    root = Path("artifacts/revision-v6")
    gate = json.loads((root / "quality-gate.json").read_text())
    frozen = json.loads((root / "frozen-selection.json").read_text())
    measured = json.loads((root / "validation.json").read_text())
    assert gate["passed"] and gate["targets"] == load_quality_targets()
    tests = ET.parse("artifacts/current-tests.xml").getroot()
    cases = tests.findall(".//testcase")
    assert len(cases) >= 78
    assert all(
        not any(c.find(tag) is not None for tag in ("failure", "error", "skipped")) for c in cases
    )
    assert any(c.attrib["name"] == "test_stage11_meets_all_five_chunk_targets" for c in cases)
    cfg = replace(load_config(root / "selected.yaml"), state="artifacts/index-current")
    source = Path(cfg.dataset) / frozen["validation_file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == frozen["validation_hash"]
    assert measured["labels_hash"] == frozen["validation_hash"]
    store = QdrantStore(path=str(Path(cfg.state) / "qdrant"))
    try:
        snapshot = build_index(cfg, SentenceEmbedder(cfg), store)
        assert snapshot["corpus_hash"] == frozen["corpus_hash"]
        assert snapshot["chunk_hash"] == frozen["chunk_hash"]
    finally:
        store.close()
    default = Path("config/default.yaml")
    baseline = Path("config/baseline.yaml")
    if not baseline.exists():
        baseline.write_bytes(default.read_bytes())
    tmp = default.with_name(f".default-{uuid4().hex}.tmp")
    tmp.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    tmp.replace(default)
    summary = measured["summary"]["hybrid+rerank"]
    atomic_json(
        Path("docs/results/current-quality.json"),
        {
            "passed": True,
            "config": asdict(cfg),
            "split": measured["split"],
            "answerable_count": summary["answerable_count"],
            "no_answer_count": len(summary["no_answer_queries"]),
            "targets": load_quality_targets(),
            "chunk_metrics": summary["chunk"],
            "latency_ms_p95": summary["latency_ms_p95"],
            "software_tests_passed": len(cases),
            "label_hash": frozen["validation_hash"],
            "corpus_hash": frozen["corpus_hash"],
            "chunk_hash": frozen["chunk_hash"],
            "query_split_only": True,
            "source_facts_overlap_development": True,
        },
    )
    print(format_target_table(summary))


if __name__ == "__main__":
    main()
