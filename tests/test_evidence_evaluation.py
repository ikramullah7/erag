"""Evidence coverage must preserve source boundaries and fact denominators."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from erag.evidence_evaluation import (
    analyze_evidence,
    contains_span,
    fact_coverage,
    quote_spans,
    validate_pool,
    verify_frozen_files,
)


@pytest.fixture
def reports():
    text = "red green blue amber"
    doc = {"id": "d", "version": "v", "text": text}
    chunks = [
        {"id": cid, "document_id": "d", "version": "v", "start": a, "end": b, "text": text[a:b]}
        for cid, a, b in (("c1", 0, 10), ("c2", 10, 20))
    ]
    snapshot = {
        "corpus_hash": "corpus",
        "chunk_hash": "chunks",
        "documents": [doc],
        "chunks": chunks,
    }

    def span(needle):
        a = text.index(needle)
        return {
            "document_id": "d",
            "version": "v",
            "start": a,
            "end": a + len(needle),
            "text": needle,
        }

    pool = {
        "version": "test",
        "status": "proposal",
        "exhaustive": False,
        "corpus_hash": "corpus",
        "cases": [
            {
                "id": "q",
                "query": "colors?",
                "facts": [
                    {
                        "id": "f1",
                        "alternatives": [{"spans": [span("red")]}, {"spans": [span("amber")]}],
                    },
                    {"id": "f2", "alternatives": [{"spans": [span("green"), span("blue")]}]},
                ],
                "relevance_evidence": [{"document_id": "d", "text": "red", "grade": 3}],
            }
        ],
    }
    ranking = {
        "corpus_hash": "corpus",
        "chunk_hash": "chunks",
        "queries": [
            {
                "case_id": "q",
                "mode": "hybrid",
                "reranked": True,
                "diagnostics": {"query": "colors?", "final_ranking": [("c1", 1), ("c2", 0.5)]},
                "chunk_metrics": {"hit@1": 1},
            }
        ],
    }
    answers = {
        "corpus_hash": "corpus",
        "chunk_hash": "chunks",
        "cases": [
            {
                "id": "q",
                "query": "colors?",
                "result": {
                    "sources": [
                        {"source_id": "S1", "chunk": {**chunks[0], "end": 20, "text": text}}
                    ],
                    "catalog": [{"source_id": "S1", "quote": "red"}],
                    "claims": [{"citations": [{"source_id": "S1", "quote": "red"}]}],
                },
            }
        ],
    }
    return pool, snapshot, ranking, answers


def test_raw_expanded_visible_and_selected_are_separate(reports):
    report = analyze_evidence(*reports)
    s = report["summary"]
    assert s["raw_top1"]["macro_fact_recall"] == 0.5
    assert s["raw_top3"]["all_facts_present_rate"] == 1
    assert s["expanded_context"]["macro_fact_recall"] == 1
    assert s["model_visible"]["macro_fact_recall"] == 0.5
    assert s["selected_quotes"]["missing_case_ids"] == ["q"]
    assert report["cases"][0]["original_raw_chunk_metrics"] == {"hit@1": 1}


def test_post_generation_context_is_not_counted_as_model_input(reports):
    result = reports[3]["cases"][0]["result"]
    full = deepcopy(result["sources"][0])
    full["source_id"] = "SC1"
    result["sources"][0]["chunk"].update(end=3, text="red")
    result["sources"].append(full)
    result["model_source_ids"] = ["S1"]
    result["claims"] = [{"citations": [{"source_id": "SC1", "quote": "red green blue amber"}]}]
    summary = analyze_evidence(*reports)["summary"]
    assert summary["expanded_context"]["macro_fact_recall"] == 0.5
    assert summary["model_visible"]["macro_fact_recall"] == 0.5
    assert summary["selected_quotes"]["macro_fact_recall"] == 1
    result["model_source_ids"].append("unknown")
    with pytest.raises(ValueError, match="identifiers"):
        analyze_evidence(*reports)


def test_equivalent_witnesses_are_or_not_extra_required_facts(reports):
    facts = reports[0]["cases"][0]["facts"]
    assert fact_coverage(facts, [("d", 15, 20)])["fact_recall"] == 0.5
    # Two different witnesses for the same requirement still earn only one fact.
    assert fact_coverage(facts, [("d", 0, 3), ("d", 15, 20)])["fact_recall"] == 0.5
    assert fact_coverage(facts, [("d", 4, 9)])["fact_recall"] == 0


def test_union_of_chunks_covers_cross_boundary_span_but_not_gaps():
    span = {"document_id": "d", "start": 0, "end": 9}
    assert contains_span([("d", 4, 9), ("d", 0, 4)], span)
    assert not contains_span([("d", 0, 4), ("d", 5, 9)], span)
    assert not contains_span([("d", 0, 4), ("other", 4, 9)], span)


def test_control_without_requirements_is_not_perfect_coverage():
    assert fact_coverage([], []) == {"fact_recall": None, "all_facts_present": None, "facts": []}


def test_provider_error_without_trace_is_unknown_context_and_zero_final_coverage(reports):
    reports[3]["cases"][0]["result"] = {"status": "provider_error", "claims": []}
    summary = analyze_evidence(*reports)["summary"]
    assert summary["model_visible"]["macro_fact_recall"] is None
    assert summary["model_visible"]["unmeasured_case_ids"] == ["q"]
    assert summary["model_visible"]["questions_with_requirements"] == 1
    assert summary["model_visible"]["conservative_fact_recall_including_unmeasured_as_zero"] == 0
    assert summary["selected_quotes"]["macro_fact_recall"] == 0


def test_source_corruption_and_wrong_corpus_are_rejected(reports):
    pool, snapshot, _, _ = reports
    pool["cases"][0]["facts"][0]["alternatives"][0]["spans"][0]["text"] = "fake"
    with pytest.raises(ValueError, match="witness"):
        validate_pool(pool, snapshot)
    pool["corpus_hash"] = "other"
    with pytest.raises(ValueError, match="different corpus"):
        validate_pool(pool, snapshot)


def test_quotes_require_valid_source_slice_and_binding(reports):
    _, snapshot, _, answers = reports
    sources = answers["cases"][0]["result"]["sources"]
    docs = {d["id"]: d for d in snapshot["documents"]}
    with pytest.raises(ValueError, match="outside"):
        quote_spans([{"source_id": "S2", "quote": "red"}], sources, docs)
    with pytest.raises(ValueError, match="duplicate"):
        quote_spans([], sources * 2, docs)
    with pytest.raises(ValueError, match="metadata"):
        quote_spans([{"source_id": "S1", "quote": "red", "document_id": "other"}], sources, docs)
    with pytest.raises(ValueError, match="metadata"):
        quote_spans([{"source_id": "S1", "quote": "red", "start": 5}], sources, docs)
    sources[0]["chunk"]["version"] = "old"
    with pytest.raises(ValueError, match="stale"):
        quote_spans([], sources, docs)


def test_duplicate_cases_and_context_mismatches_cannot_silently_align(reports):
    args = deepcopy(reports)
    args[3]["cases"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        analyze_evidence(*args)
    reports[3]["cases"][0]["query"] = "other question"
    with pytest.raises(ValueError, match="question/context"):
        analyze_evidence(*reports)


def test_empty_pool_and_changed_frozen_inputs_fail_closed(reports, tmp_path):
    pool, snapshot, _, _ = reports
    pool["cases"] = []
    with pytest.raises(ValueError, match="requires cases"):
        validate_pool(pool, snapshot)
    path = tmp_path / "config"
    path.write_text("before")
    manifest = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    verify_frozen_files(manifest)
    path.write_text("after")
    with pytest.raises(ValueError, match="frozen input changed"):
        verify_frozen_files(manifest)


@pytest.mark.skipif(
    not Path("artifacts/real-world/stage1/answers-v2-grounded-final/answers.json").exists(),
    reason="requires preserved local development artifacts",
)
def test_real_trace_separates_label_omissions_expansion_and_answer_selection():
    paths = (
        "data/real-world/stage1/relevance-review-v2.json",
        "artifacts/real-world/stage1/index-v2/manifest.json",
        "artifacts/real-world/stage1/run-002/heldout.json",
        "artifacts/real-world/stage1/answers-v2-grounded-final/answers.json",
    )
    inputs = [json.loads(Path(p).read_text()) for p in paths]
    before = deepcopy(inputs)
    r = analyze_evidence(*inputs)
    assert inputs == before
    cases = {c["id"]: c for c in r["cases"]}
    assert cases["q05"]["original_raw_chunk_metrics"]["hit@1"] == 0
    assert cases["q05"]["reviewed_pool_raw_chunk_metrics"]["hit@1"] == 1
    assert cases["q14"]["coverage"]["raw_top5"]["fact_recall"] == pytest.approx(2 / 3)
    assert cases["q14"]["coverage"]["model_visible"]["all_facts_present"]
    assert cases["q13"]["coverage"]["model_visible"]["all_facts_present"]
    assert not cases["q13"]["coverage"]["selected_quotes"]["all_facts_present"]
