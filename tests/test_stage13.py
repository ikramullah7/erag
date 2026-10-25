"""Twenty-document stage boundaries, retained identities and frozen evidence validation."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import FakeEmbedder

from erag.answer_evaluation import load_answer_cases
from erag.chunking import chunk_document
from erag.config import load_config
from erag.evaluation import load_cases
from erag.ingestion import STAGE_12_IDS, STAGE_13_IDS, ingest
from erag.pipeline import Engine, build_index
from erag.storage import QdrantStore


@pytest.fixture
def stage13(tmp_path):
    source = tmp_path / "sources"
    shutil.copytree("data/stage-1.3", source)
    return replace(
        load_config("config/stage-1.3.yaml"), dataset=str(source), state=str(tmp_path / "index")
    )


def test_twenty_sources_retain_every_prior_source_and_chunk_identity():
    cfg = load_config("config/stage-1.3.yaml")
    for path in Path("data/stage-1.2").glob("*.txt"):
        assert path.read_bytes() == (Path(cfg.dataset) / path.name).read_bytes()
    assert {d.id for d in ingest(cfg.dataset, "1.3")} == STAGE_13_IDS
    old = [
        c.id
        for d in ingest("data/stage-1.2", "1.2")
        for c in chunk_document(d, cfg.max_words, cfg.overlap_words, cfg.section_boundaries)
    ]
    new = [
        c.id
        for d in ingest(cfg.dataset, "1.3")
        if d.id in STAGE_12_IDS
        for c in chunk_document(d, cfg.max_words, cfg.overlap_words, cfg.section_boundaries)
    ]
    assert old == new and len(old) == 52
    with pytest.raises(ValueError):
        ingest(cfg.dataset, "1.2")
    with pytest.raises(ValueError):
        ingest(cfg.dataset, "1.4")
    with pytest.raises(ValueError):
        replace(cfg, stage="1.4")


def test_stage13_rejects_additional_and_missing_sources(stage13):
    p = Path(stage13.dataset) / "unapproved.txt"
    p.write_text("Unexpected twenty-first source")
    with pytest.raises(ValueError):
        ingest(stage13.dataset, "1.3")
    p.unlink()
    (Path(stage13.dataset) / "servicebus-recovery.txt").unlink()
    with pytest.raises(ValueError):
        ingest(stage13.dataset, "1.3")


def test_twenty_document_index_deletion_restore_and_stale_source(stage13):
    store = QdrantStore()
    embedder = FakeEmbedder()
    try:
        initial = build_index(stage13, embedder, store)
        assert len(initial["documents"]) == 20 and len(initial["chunks"]) == 82
        engine = Engine(stage13, store, embedder)
        assert (
            engine.search("payments-events-prod PeekLock", mode="bm25")["results"][0]["chunk"][
                "document_id"
            ]
            == "servicebus-recovery"
        )
        removed = build_index(stage13, embedder, store, ["vault-rotation", "aks-platform"])
        assert not any(
            c["document_id"] in {"vault-rotation", "aks-platform"} for c in removed["chunks"]
        )
        Engine(stage13, store, embedder)
        assert build_index(stage13, embedder, store, [])["chunk_hash"] == initial["chunk_hash"]
        path = Path(stage13.dataset) / "patch-compliance.txt"
        path.write_text(path.read_text() + "\nNew source version.\n")
        with pytest.raises(ValueError, match="changed"):
            Engine(stage13, store, embedder)
    finally:
        store.close()


def test_stage13_question_splits_and_evidence_are_complete(stage13):
    store = QdrantStore()
    embedder = FakeEmbedder()
    try:
        build_index(stage13, embedder, store)
        engine = Engine(stage13, store, embedder)
        cases = load_answer_cases(Path(stage13.dataset) / "answer-evaluation-v1.json", engine)
        assert len(cases) == 36
        assert sum(c["group"] == "legacy-regression" for c in cases) == 10
        assert sum(c["expected_status"] == "insufficient_evidence" for c in cases) == 6
        tuning = load_cases(Path(stage13.dataset) / "tuning.json", engine)
        heldout = load_cases(Path(stage13.dataset) / "heldout.json", engine)
        assert len(tuning) == 15 and len(heldout) == 21
        assert {c["id"] for c in tuning}.isdisjoint(c["id"] for c in heldout)
        refs = json.loads(
            (Path(stage13.dataset) / "answer-semantic-references-v1.json").read_text()
        )
        assert [
            {k: v for k, v in c.items() if k != "required_facts"} for c in refs["cases"]
        ] == cases
    finally:
        store.close()


def test_reviewed_twenty_document_baseline_keeps_semantic_findings_visible(tmp_path):
    from erag.reviewed_metrics import score_reviewed_answers

    report = score_reviewed_answers(
        "docs/results/stage-1.3/answers.json.gz",
        "data/stage-1.3/answer-semantic-references-v1.json",
        "data/stage-1.3/answer-reviewed-baseline-v1.json",
        tmp_path / "baseline.json",
    )
    assert report["phase"] == "answer-1.3" and report["documents"] == 20
    assert len(report["cases"]) == 36
    assert report["summary"]["faithfulness_counts"] == {"supported": 71, "total": 74}
    assert report["summary"]["citation_accuracy_counts"] == {"supported": 67, "total": 70}
    assert report["summary"]["answer_correctness"] == pytest.approx(0.9625252525252525)
    assert report["summary"]["abstention_accuracy"] == 1
    rows = {c["id"]: c for c in report["cases"]}
    assert (
        rows["fresh-07"]["judgment"]["claims"][2]["atomic_facts"][1]["faithfulness"]
        == "unsupported"
    )
    assert not rows["fresh-10"]["judgment"]["required_facts"][0]["covered"]
    assert rows["fresh-19"]["judgment"]["claims"][1]["citations"][0]["support"] == "unsupported"
