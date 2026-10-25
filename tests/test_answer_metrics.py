"""Semantic scoring must catch omissions, bad citations, incomplete answers and judge failures."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from erag.answer_config import load_answer_config
from erag.answer_metrics import (
    calibrate_judge,
    citation_provenance,
    evaluate_semantics,
    format_metrics,
    judge_answer,
    read_json,
    score_judgment,
    summarize,
    validate_judgment,
)
from erag.answering import Generation


def judgment(claims, facts):
    return {
        "claims": [
            {
                "claim_index": i,
                "atomic_facts": [
                    {
                        "text": c["text"],
                        "faithfulness": "supported",
                        "correctness": "correct",
                        "reason": "Direct evidence.",
                    }
                ],
                "citations": [
                    {"citation_index": j, "support": "supported", "reason": "Supports claim."}
                    for j, _ in enumerate(c["citations"])
                ],
            }
            for i, c in enumerate(claims)
        ],
        "required_facts": [
            {"fact_index": i, "covered": True, "reason": "Present."} for i, _ in enumerate(facts)
        ],
    }


def task():
    return {
        "question": "CPU and memory?",
        "required_facts": ["500m CPU", "768Mi memory"],
        "reference_evidence": [],
        "retrieved_excerpts": [],
        "claims": [{"text": "500m CPU", "citations": [{"quote": "500m CPU", "source_id": "S1"}]}],
    }


class AlwaysPositiveJudge:
    def generate(self, messages, schema):
        import json

        value = json.loads(messages[1]["content"])
        return Generation(judgment(value["claims"], value["required_facts"]), "test-judge", {}, 1)


@pytest.mark.parametrize("change", ["claim", "citation", "reference", "boolean"])
def test_missing_or_duplicate_judgments_are_rejected(change):
    t = task()
    value = judgment(t["claims"], t["required_facts"])
    if change == "claim":
        value["claims"][0]["claim_index"] = 1
    elif change == "citation":
        value["claims"][0]["citations"] = []
    elif change == "reference":
        value["required_facts"][1]["fact_index"] = 0
    else:
        value["required_facts"][0]["covered"] = "true"
    with pytest.raises(ValueError):
        validate_judgment(value, t["claims"], t["required_facts"])


def test_partial_correct_answer_loses_completeness_without_losing_faithfulness():
    t = task()
    value = judgment(t["claims"], t["required_facts"])
    value["required_facts"][1]["covered"] = False
    s = score_judgment(value, [[True]])
    assert s["supported_atomic_facts"] == s["atomic_facts"] == 1
    assert s["correctness_precision"] == 1 and s["correctness_recall"] == 0.5
    assert s["answer_correctness_f1"] == pytest.approx(2 / 3)


def test_authentic_but_unrelated_quote_fails_semantic_citation_accuracy():
    t = task()
    value = judgment(t["claims"], t["required_facts"])
    value["claims"][0]["citations"][0]["support"] = "unsupported"
    s = score_judgment(value, [[True]])
    assert s["citations"] == 1 and s["supported_citations"] == 0
    # Conversely, a positive semantic verdict cannot rescue invalid source provenance.
    value["claims"][0]["citations"][0]["support"] = "supported"
    assert score_judgment(value, [[False]])["supported_citations"] == 0


def test_judge_failure_never_disappears_from_aggregate():
    rows = [
        {
            "id": "a",
            "expected_status": "answered",
            "actual_status": "answered",
            "abstention_decision_correct": True,
            "judge_error": "timeout",
        },
        {
            "id": "b",
            "expected_status": "insufficient_evidence",
            "actual_status": "insufficient_evidence",
            "abstention_decision_correct": True,
        },
    ]
    s = summarize(rows)
    assert s["faithfulness"] is None and s["answer_correctness"] is None
    assert s["citation_accuracy"] is None and s["abstention_accuracy"] == 1
    assert not s["semantic_scoring_complete"] and "Unscored" in format_metrics({"summary": s})


def test_abstention_accuracy_includes_false_abstentions_and_provider_errors():
    rows = [
        {
            "id": "a",
            "expected_status": "answered",
            "actual_status": "insufficient_evidence",
            "abstention_decision_correct": False,
        },
        {
            "id": "b",
            "expected_status": "insufficient_evidence",
            "actual_status": "provider_error",
            "abstention_decision_correct": False,
        },
    ]
    assert summarize(rows)["abstention_accuracy"] == 0
    assert summarize(rows)["false_abstentions"] == 1


def test_modified_retrieved_context_is_rejected_before_judging():
    chunk = {"document_id": "doc", "version": "v", "text": "altered", "start": 0, "end": 7}
    result = {"sources": [{"source_id": "S1", "chunk": chunk}], "claims": []}
    doc = SimpleNamespace(text="original", version="v")
    with pytest.raises(ValueError, match="modified or stale"):
        citation_provenance(result, {"doc": doc})


def test_always_positive_judge_fails_negative_calibration_cases(tmp_path):
    r = calibrate_judge(
        AlwaysPositiveJudge(),
        load_answer_config("config/answer-judge.yaml"),
        "data/stage-1.1/semantic-calibration-v1.json",
        tmp_path / "cal.json",
    )
    assert not r["passed"] and any(not c["passed"] for c in r["cases"])


def test_semantic_pipeline_accepts_archived_answers_and_preserves_prior_output(tmp_path):
    output = tmp_path / "semantic.json"
    cfg = load_answer_config("config/answer-judge.yaml")
    args = (
        "docs/results/answers-phase-1.1/v2-qwen3-14b.json.gz",
        "data/stage-1.1/answer-semantic-references-v1.json",
        AlwaysPositiveJudge(),
        cfg,
        output,
    )
    r = evaluate_semantics(*args)
    assert r["summary"]["semantic_scoring_complete"] and len(r["cases"]) == 20
    assert r["summary"]["abstention_accuracy"] == 1
    with pytest.raises(ValueError, match="preserve"):
        evaluate_semantics(*args)


def test_oversized_judge_context_is_rejected_without_truncation():
    t = deepcopy(task())
    t["question"] = "x" * 20000
    with pytest.raises(ValueError, match="context budget"):
        judge_answer(AlwaysPositiveJudge(), load_answer_config("config/answer-judge.yaml"), t)


def test_original_questions_and_evidence_unchanged_in_semantic_references():
    old = read_json("data/stage-1.1/answer-evaluation-v1.json")
    new = read_json("data/stage-1.1/answer-semantic-references-v1.json")
    assert [{k: v for k, v in c.items() if k != "required_facts"} for c in new["cases"]] == old[
        "cases"
    ]


def test_correctness_and_faithfulness_use_different_evidence():
    t = task()
    value = judgment(t["claims"], t["required_facts"])
    value["claims"][0]["atomic_facts"][0]["faithfulness"] = "unsupported"
    s = score_judgment(value, [[True]])
    assert s["supported_atomic_facts"] == 0
    assert s["correct_atomic_facts"] == 1 and s["answer_correctness_f1"] == 1


def test_coverage_pass_cannot_credit_reference_evidence_as_answer_text():
    import json

    class CoverageJudge:
        calls = None

        def __init__(self):
            self.calls = []

        def generate(self, messages, schema):
            t = json.loads(messages[1]["content"])
            self.calls.append(t)
            value = judgment(t["claims"], t["required_facts"])
            if len(self.calls) == 2:
                value["claims"][0]["atomic_facts"][0]["faithfulness"] = "unsupported"
                value["claims"][0]["atomic_facts"][0]["correctness"] = "unknown"
            return Generation(value, "test-judge", {}, 1)

    t = task()
    t["reference_evidence"] = [{"text": "REFERENCE_ONLY_SENTINEL"}]
    t["retrieved_excerpts"] = [{"source_id": "S1", "text": "REFERENCE_ONLY_SENTINEL"}]
    t["required_facts"] = ["A replica requests 768Mi memory."]
    evaluator = CoverageJudge()
    r = judge_answer(evaluator, load_answer_config("config/answer-judge.yaml"), t)
    assert r["source_pass"]["required_facts"][0]["covered"] is True
    assert r["judgment"]["required_facts"][0]["covered"] is False
    assert len(evaluator.calls) == 2
    assert "REFERENCE_ONLY_SENTINEL" not in json.dumps(evaluator.calls[1])
    assert evaluator.calls[1]["retrieved_excerpts"][0]["text"] == "500m CPU"
    assert r["elapsed_ms"] == 2


def test_abstention_is_available_without_claiming_unjudged_semantic_scores():
    from erag.answer_metrics import abstention_only

    report = abstention_only(
        "docs/results/answers-phase-1.1/v2-qwen3-14b.json.gz",
        "data/stage-1.1/answer-semantic-references-v1.json",
    )
    assert report["summary"]["abstention_accuracy"] == 1
    assert report["summary"]["unsupported_abstention_recall"] == 1
    assert report["summary"]["answer_correctness"] is None
    assert not report["summary"]["semantic_scoring_complete"]
