import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import FakeReranker
from qdrant_client import models

from erag.config import load_config
from erag.evaluation import load_cases
from erag.pipeline import Engine, build_index


def test_all_retrieval_modes_and_reranking(indexed):
    cfg, store, embedder, _snapshot = indexed
    engine = Engine(cfg, store, embedder, FakeReranker())
    for mode in ("bm25", "vector", "hybrid"):
        for rerank in (False, True):
            r = engine.search("certificate payments-api-tls", mode, rerank)
            assert r["results"]
            assert all(x["chunk"]["id"] in engine.by_id for x in r["results"])
            assert {x[0] for x in r["final_ranking"]} == {x[0] for x in r["initial_ranking"]}
            assert all("cross_encoder" in x["scores"] for x in r["results"])
    assert (
        engine.search("10.42.8.17", "bm25")["results"][0]["chunk"]["document_id"]
        == "azure-networking"
    )


def test_rebuild_is_idempotent_and_old_snapshot_survives(indexed):
    cfg, store, embedder, old = indexed
    new = build_index(cfg, embedder, store)
    assert old["chunks"] == new["chunks"]
    assert old["collection"] != new["collection"]
    store.validate(old["collection"], Engine(cfg, store, embedder).chunks)


def test_update_replaces_old_chunks_and_preserves_document_id(indexed):
    cfg, store, embedder, old = indexed
    source = Path(cfg.dataset) / "aks-platform.txt"
    source.write_text(source.read_text() + "\n\nA new upgrade note.\n")
    with pytest.raises(ValueError, match="changed"):
        Engine(cfg, store, embedder)
    new = build_index(cfg, embedder, store)
    old_ids = {c["id"] for c in old["chunks"] if c["document_id"] == "aks-platform"}
    new_ids = {c["id"] for c in new["chunks"] if c["document_id"] == "aks-platform"}
    assert old_ids.isdisjoint(new_ids)
    assert {c["document_id"] for c in old["chunks"]} == {c["document_id"] for c in new["chunks"]}
    Engine(cfg, store, embedder)


def test_removed_document_fails_stage_gate_without_changing_manifest(indexed):
    cfg, store, embedder, old = indexed
    (Path(cfg.dataset) / "aks-platform.txt").unlink()
    with pytest.raises(ValueError):
        build_index(cfg, embedder, store)
    assert (
        json.loads((Path(cfg.state) / "manifest.json").read_text())["collection"]
        == old["collection"]
    )


def test_partial_build_failure_preserves_active_snapshot(indexed):
    cfg, store, embedder, old = indexed

    class Broken:
        identity = "broken"

        def encode(self, texts):
            raise RuntimeError("injected model failure")

    with pytest.raises(RuntimeError):
        build_index(cfg, Broken(), store)
    assert (
        json.loads((Path(cfg.state) / "manifest.json").read_text())["collection"]
        == old["collection"]
    )
    Engine(cfg, store, embedder)


def test_missing_qdrant_point_rejected(indexed):
    cfg, store, embedder, snapshot = indexed
    store.client.delete(
        snapshot["collection"],
        points_selector=models.PointIdsList(points=[snapshot["chunks"][0]["id"]]),
        wait=True,
    )
    with pytest.raises(ValueError, match="mismatch"):
        Engine(cfg, store, embedder)


def test_modified_payload_rejected(indexed):
    cfg, store, embedder, snapshot = indexed
    store.client.set_payload(
        snapshot["collection"],
        payload={"text": "corrupt"},
        points=[snapshot["chunks"][0]["id"]],
        wait=True,
    )
    with pytest.raises(ValueError, match="mismatch"):
        Engine(cfg, store, embedder)


def test_changed_chunk_config_rejected(indexed):
    cfg, store, embedder, _ = indexed
    with pytest.raises(ValueError, match="configuration"):
        Engine(replace(cfg, max_words=180), store, embedder)


def test_model_mismatch_rejected(indexed):
    cfg, store, embedder, _ = indexed
    embedder.identity = "different"
    with pytest.raises(ValueError, match="identity"):
        Engine(cfg, store, embedder)


def test_tuning_evidence_grounded_and_covered(indexed):
    cfg, store, embedder, _ = indexed
    cases = load_cases(Path(cfg.dataset) / "tuning.json", Engine(cfg, store, embedder))
    assert len(cases) == 18
    assert sum(bool(c["relevant_chunks"]) for c in cases) == 15


@pytest.mark.parametrize("size", [100, 140, 180])
def test_tuning_coverage_across_chunk_configurations(indexed, size):
    cfg, store, embedder, _ = indexed
    cfg = replace(cfg, max_words=size)
    build_index(cfg, embedder, store)
    load_cases(Path(cfg.dataset) / "tuning.json", Engine(cfg, store, embedder))


def test_invalid_embedding_matrix_rejected(indexed):
    cfg, store, embedder, _ = indexed
    engine = Engine(cfg, store, embedder)
    with pytest.raises(ValueError):
        store.build("bad", engine.chunks, np.full((len(engine.chunks), 3), np.nan))


@pytest.mark.parametrize("query", ["", " ", "x" * 4001])
def test_invalid_query(indexed, query):
    cfg, store, embedder, _ = indexed
    with pytest.raises(ValueError):
        Engine(cfg, store, embedder).search(query)


def test_unknown_config_key(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("typo: true")
    with pytest.raises(ValueError, match="unknown"):
        load_config(p)


def test_delete_removes_from_both_indexes_and_restore(indexed):
    cfg, store, embedder, original = indexed
    deleted = build_index(cfg, embedder, store, ["certificate-management"])
    engine = Engine(cfg, store, embedder)
    assert all(c.document_id != "certificate-management" for c in engine.chunks)
    assert len(engine.snapshot["documents"]) == 5
    assert len({c.document_id for c in engine.chunks}) == 4
    assert build_index(cfg, embedder, store)["deleted_documents"] == ["certificate-management"]
    restored = build_index(cfg, embedder, store, [])
    assert original["chunk_hash"] == restored["chunk_hash"]
    assert deleted["chunk_hash"] != restored["chunk_hash"]
    Engine(cfg, store, embedder)


def test_unsupported_evidence_rejected(indexed, tmp_path):
    cfg, store, embedder, _ = indexed
    p = tmp_path / "cases.json"
    cases = json.loads((Path(cfg.dataset) / "tuning.json").read_text())
    cases[0]["evidence"][0]["text"] = "A fabricated unsupported answer"
    p.write_text(json.dumps(cases))
    with pytest.raises(ValueError, match="unsupported evidence"):
        load_cases(p, Engine(cfg, store, embedder))


@pytest.mark.parametrize(
    "change", [{"mode": []}, {"embedding_revision": True}, {"qdrant_url": 123}]
)
def test_malformed_config_types_rejected(config, change):
    with pytest.raises(ValueError):
        replace(config, **change)


def test_selected_strategy_evaluation_keeps_hit_metrics_and_excludes_no_answer_from_means(
    indexed, tmp_path
):
    from erag.evaluation import evaluate

    cfg, store, embedder, _ = indexed
    report = evaluate(
        Engine(cfg, store, embedder, FakeReranker()),
        Path(cfg.dataset) / "tuning.json",
        tmp_path / "evaluation.json",
        "unit-selected-strategy",
        "tuning",
        strategies=(("hybrid", True),),
    )
    assert list(report["summary"]) == ["hybrid+rerank"]
    summary = report["summary"]["hybrid+rerank"]
    assert summary["answerable_count"] == 15
    assert len(summary["no_answer_queries"]) == 3
    assert {"hit@1", "hit@3", "recall@5", "mrr", "ndcg@10"} <= summary["chunk"].keys()


@pytest.mark.parametrize("extra_args,expected", [([], True), (["--no-rerank"], False)])
def test_cli_uses_profile_reranking_and_allows_explicit_disable(
    indexed, monkeypatch, capsys, extra_args, expected
):
    import sys

    from erag import cli

    cfg, store, embedder, _ = indexed
    cfg = replace(cfg, rerank=True)
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "QdrantStore", lambda **kwargs: store)
    monkeypatch.setattr(cli, "SentenceEmbedder", lambda config: embedder)
    monkeypatch.setattr(cli, "CrossEncoderReranker", lambda config: FakeReranker())
    monkeypatch.setattr(
        sys, "argv", ["erag", "search", "certificate", "--mode", "bm25", *extra_args]
    )
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["reranked"] is expected
