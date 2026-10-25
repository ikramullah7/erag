"""Ten-source stage isolation, retention, stable IDs, corpus updates and frozen evidence."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import FakeEmbedder, FakeReranker

from erag.answer_evaluation import load_answer_cases
from erag.chunking import chunk_document
from erag.config import load_config
from erag.evaluation import load_cases
from erag.ingestion import APPROVED_IDS, STAGE_12_IDS, ingest
from erag.pipeline import Engine, build_index
from erag.storage import QdrantStore


@pytest.fixture
def stage12(tmp_path):
    source = tmp_path / "documents"
    shutil.copytree("data/stage-1.2", source)
    return replace(
        load_config("config/stage-1.2.yaml"), dataset=str(source), state=str(tmp_path / "index")
    )


def test_original_documents_retained_byte_for_byte_and_stage_is_explicit():
    for name in APPROVED_IDS:
        assert (
            Path(f"data/stage-1.1/{name}.txt").read_bytes()
            == Path(f"data/stage-1.2/{name}.txt").read_bytes()
        )
    assert {d.id for d in ingest("data/stage-1.2", "1.2")} == STAGE_12_IDS
    with pytest.raises(ValueError):
        ingest("data/stage-1.2")
    with pytest.raises(ValueError):
        ingest("data/stage-1.1", "1.2")
    with pytest.raises(ValueError):
        ingest("data/stage-1.2", "1.3")


def test_later_or_incomplete_document_sets_rejected(stage12):
    root = Path(stage12.dataset)
    (root / "extra-document.txt").write_text("Unapproved eleventh source")
    with pytest.raises(ValueError):
        ingest(root, "1.2")
    (root / "extra-document.txt").unlink()
    (root / "aks-platform.txt").unlink()
    with pytest.raises(ValueError):
        ingest(root, "1.2")


def test_retained_chunk_identifiers_survive_stage_expansion():
    cfg = load_config("config/stage-1.2.yaml")
    old = [
        c.id
        for d in ingest("data/stage-1.1")
        for c in chunk_document(d, cfg.max_words, cfg.overlap_words, cfg.section_boundaries)
    ]
    new = [
        c.id
        for d in ingest("data/stage-1.2", "1.2")
        if d.id in APPROVED_IDS
        for c in chunk_document(d, cfg.max_words, cfg.overlap_words, cfg.section_boundaries)
    ]
    assert old == new
    assert cfg.fingerprint != replace(cfg, stage="1.1").fingerprint


def test_ten_document_index_deletion_restore_and_source_change(stage12):
    store = QdrantStore()
    embedder = FakeEmbedder()
    try:
        initial = build_index(stage12, embedder, store)
        assert len(initial["documents"]) == 10 and len(initial["chunks"]) == 52
        engine = Engine(stage12, store, embedder, FakeReranker())
        assert engine.search("PostgreSQL backup retention", mode="bm25")["results"]
        removed = build_index(stage12, embedder, store, ["redis-availability"])
        assert not any(c["document_id"] == "redis-availability" for c in removed["chunks"])
        Engine(stage12, store, embedder)
        restored = build_index(stage12, embedder, store, [])
        assert restored["chunk_hash"] == initial["chunk_hash"]
        with pytest.raises(ValueError):
            build_index(stage12, embedder, store, ["unapproved"])
        file = Path(stage12.dataset) / "redis-availability.txt"
        file.write_text(file.read_text() + "\nNew source version.\n")
        with pytest.raises(ValueError, match="changed"):
            Engine(stage12, store, embedder)
    finally:
        store.close()


def test_frozen_answer_and_ranking_labels_are_source_grounded(stage12):
    store = QdrantStore()
    embedder = FakeEmbedder()
    try:
        build_index(stage12, embedder, store)
        engine = Engine(stage12, store, embedder)
        assert (
            len(load_answer_cases(Path(stage12.dataset) / "answer-evaluation-v1.json", engine))
            == 36
        )
        sets = []
        for split in ("tuning", "heldout"):
            sets.append(
                {c["id"] for c in load_cases(Path(stage12.dataset) / (split + ".json"), engine)}
            )
        assert not sets[0] & sets[1] and len(sets[0] | sets[1]) == 36
        cases = json.loads((Path(stage12.dataset) / "answer-evaluation-v1.json").read_text())[
            "cases"
        ]
        assert sum(c["group"] == "legacy-regression" for c in cases) == 15
        assert sum(c["expected_status"] == "insufficient_evidence" for c in cases) == 6
    finally:
        store.close()


def test_reviewed_baseline_scores_the_ten_document_stage_without_a_judge(tmp_path):
    from erag.reviewed_metrics import score_reviewed_answers

    report = score_reviewed_answers(
        "docs/results/stage-1.2/answers.json.gz",
        "data/stage-1.2/answer-semantic-references-v1.json",
        "data/stage-1.2/answer-reviewed-baseline-v1.json",
        tmp_path / "reviewed.json",
    )
    assert report["phase"] == "answer-1.2" and report["documents"] == 10
    assert len(report["cases"]) == 36
    assert report["summary"]["faithfulness_counts"] == {"supported": 72, "total": 73}
    assert report["summary"]["citation_accuracy_counts"] == {"supported": 68, "total": 70}
    assert report["summary"]["answer_correctness"] == pytest.approx(35 / 36)
    assert report["summary"]["abstention_accuracy"] == 1
    assert any("36-case" in note for note in report["limitations"])
