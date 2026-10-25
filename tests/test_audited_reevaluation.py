"""Cached-rank re-evaluation must match fresh evaluation and reject changed inputs."""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from conftest import FakeReranker

from erag.evaluation import evaluate
from erag.pipeline import Engine

spec = importlib.util.spec_from_file_location(
    "audited_reevaluator",
    Path(__file__).resolve().parents[1] / "scripts/reevaluate_audited_labels.py",
)
reevaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reevaluator)


@pytest.fixture
def cached_run(indexed, tmp_path):
    cfg, store, embedder, _ = indexed
    engine = Engine(cfg, store, embedder, FakeReranker())
    original = Path(cfg.dataset) / "tuning.json"
    cache = tmp_path / "cached.json"
    evaluate(engine, original, cache, "cache-reference", "tuning", strategies=(("hybrid", True),))
    revised = tmp_path / "revised" / "tuning.json"
    revised.parent.mkdir()
    cases = json.loads(original.read_text())
    evidence = cases[0]["evidence"][0]
    evidence["grade"] = 2
    cases[0]["relevant_documents"][evidence["document_id"]] = 2
    revised.write_text(json.dumps(cases))
    return engine, cache, revised


def test_cached_rescore_matches_fresh_evaluation_under_revised_labels(
    cached_run, tmp_path, monkeypatch
):
    engine, cache, revised = cached_run
    fresh = evaluate(
        engine,
        revised,
        tmp_path / "fresh.json",
        "fresh-reference",
        "tuning",
        strategies=(("hybrid", True),),
    )

    def no_inference(*args, **kwargs):
        raise AssertionError("rescore must not execute retrieval")

    monkeypatch.setattr(engine, "search", no_inference)
    scored = reevaluator.rescore_cached(engine, revised, cache, "hybrid", True)
    key = "hybrid+rerank"
    assert scored["summary"][key]["chunk"] == fresh["summary"][key]["chunk"]
    assert scored["summary"][key]["document"] == fresh["summary"][key]["document"]
    assert not scored["inference_reexecuted"]
    assert [q["relevant_chunks"] for q in scored["queries"]] == [
        q["relevant_chunks"] for q in fresh["queries"]
    ]


def test_cached_rescore_rejects_changed_question_text(cached_run):
    engine, cache, revised = cached_run
    cases = json.loads(revised.read_text())
    cases[0]["query"] = "A different question"
    revised.write_text(json.dumps(cases))
    with pytest.raises(ValueError, match="query text changed", check=lambda e: True):
        reevaluator.rescore_cached(engine, revised, cache, "hybrid", True)


def test_cached_rescore_rejects_changed_ranking_configuration(cached_run):
    engine, cache, revised = cached_run
    report = json.loads(cache.read_text())
    report["config"]["bm25_weight"] = asdict(engine.cfg)["bm25_weight"] + 1
    cache.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="configuration mismatch"):
        reevaluator.rescore_cached(engine, revised, cache, "hybrid", True)
