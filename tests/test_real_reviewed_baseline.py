"""The optional frozen real pilot keeps failures and provisional review judgments visible."""

from pathlib import Path

import pytest

from erag.reviewed_metrics import score_reviewed_answers


@pytest.mark.skipif(
    not Path("artifacts/real-world/stage1/results/answers.json.gz").exists(),
    reason="local official-document acquisition/evaluation artifacts are required",
)
def test_real_reviewed_baseline_preserves_actual_errors_and_false_abstentions(tmp_path):
    r = score_reviewed_answers(
        "artifacts/real-world/stage1/results/answers.json.gz",
        "data/real-world/stage1/answer-semantic-references-v1.json",
        "data/real-world/stage1/answer-reviewed-baseline-v1.json",
        tmp_path / "baseline.json",
    )
    assert r["phase"] == "answer-real-1" and r["documents"] == 5
    assert len(r["cases"]) == 22 and r["summary"]["false_abstentions"] == 4
    assert r["summary"]["faithfulness_counts"] == {"supported": 43, "total": 46}
    assert r["summary"]["citation_accuracy_counts"] == {"supported": 34, "total": 44}
    assert r["summary"]["answer_correctness"] == pytest.approx(0.5203703703703704)
    assert r["summary"]["abstention_accuracy"] == pytest.approx(17 / 22)
    rows = {c["id"]: c for c in r["cases"]}
    assert rows["q14"]["judgment"]["claims"][0]["atomic_facts"][0]["correctness"] == "incorrect"
    assert not rows["q08"]["judgment"]["required_facts"][1]["covered"]
    assert rows["q05"]["actual_status"] == "invalid_response"
