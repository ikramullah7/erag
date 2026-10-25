"""Reviewed baselines bind annotations to frozen answers and never require an LLM."""

import json

import pytest

from erag.answer_metrics import read_json
from erag.reviewed_metrics import score_reviewed_answers

ANSWERS = "docs/results/answers-phase-1.1/v2-qwen3-14b.json.gz"
REFERENCES = "data/stage-1.1/answer-semantic-references-v1.json"
REVIEWS = "data/stage-1.1/answer-reviewed-baseline-v1.json"


def test_reviewed_baseline_counts_and_conservative_qualifier_judgment(tmp_path):
    path = tmp_path / "baseline.json"
    report = score_reviewed_answers(ANSWERS, REFERENCES, REVIEWS, path)
    s = report["summary"]
    assert s["faithfulness_counts"] == {"supported": 29, "total": 30}
    assert s["citation_accuracy_counts"] == {"supported": 28, "total": 29}
    assert s["answer_correctness"] == pytest.approx(14.5 / 15)
    assert s["abstention_accuracy"] == 1 and s["false_abstentions"] == 0
    cert = next(c for c in report["cases"] if c["id"] == "answer-04")
    assert cert["scores"]["answer_correctness_f1"] == 0.5
    assert cert["citation_provenance"] == [[True], [True]]
    assert cert["judgment"]["claims"][0]["citations"][0]["support"] == "unsupported"
    assert all(
        c["scores"]["answer_correctness_f1"] == 1
        for c in report["cases"]
        if c["expected_status"] == "answered" and c["id"] != "answer-04"
    )
    assert report["review_kind"] == "agent-reviewed" and len(report["cases"]) == 20
    with pytest.raises(ValueError, match="preserve"):
        score_reviewed_answers(ANSWERS, REFERENCES, REVIEWS, path)


@pytest.mark.parametrize("tamper", ["hash", "missing", "duplicate", "claim", "fact", "index"])
def test_changed_or_incomplete_review_cannot_replay_as_the_original_baseline(tmp_path, tamper):
    annotations = read_json(REVIEWS)
    if tamper == "hash":
        annotations["answer_input_sha256"] = "0" * 64
    elif tamper == "missing":
        annotations["cases"].pop()
    elif tamper == "duplicate":
        annotations["cases"].append(annotations["cases"][0])
    elif tamper == "claim":
        annotations["cases"][0]["answer_claims"][0] = "Modified answer"
    elif tamper == "fact":
        annotations["cases"][0]["required_facts"][0] = "Modified reference"
    else:
        annotations["cases"][0]["judgment"]["claims"][0]["claim_index"] = 99
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(annotations))
    with pytest.raises(ValueError):
        score_reviewed_answers(ANSWERS, REFERENCES, path, tmp_path / "result.json")
