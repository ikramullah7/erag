"""Real-input preparation does not invent a corpus, run a model, or bypass stage guards."""

import hashlib

import pytest

from erag.real_validation import prepare_real_validation


def questions():
    return {
        "author": "Unit-test reviewer",
        "questions": [{"id": "q1", "query": "What is the procedure?"}],
    }


def test_preparation_freezes_selected_source_and_question_without_evaluation(tmp_path):
    source = tmp_path / "documents"
    source.mkdir()
    p = source / "runbook.txt"
    p.write_text("Existing source procedure.\n")
    output = tmp_path / "manifest.json"
    result = prepare_real_validation(source, questions(), output)
    assert result["status"] == "prepared_not_indexed"
    assert result["sources"][0]["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert result["questions"] == questions()["questions"]
    assert "Not run" in result["evaluation_status"]
    assert "not independently verified" in result["independence"]
    assert p.read_text() == "Existing source procedure.\n"
    with pytest.raises(ValueError, match="preserve"):
        prepare_real_validation(source, questions(), output)


def test_duplicate_real_sources_are_rejected(tmp_path):
    (tmp_path / "first.txt").write_text("Same procedure")
    (tmp_path / "second.txt").write_text("Same procedure\n")
    with pytest.raises(ValueError, match="duplicate"):
        prepare_real_validation(tmp_path, questions(), tmp_path / "manifest.json")


def test_placeholder_questions_do_not_count_as_independent_inputs(tmp_path):
    (tmp_path / "runbook.txt").write_text("Procedure")
    with pytest.raises(ValueError, match="author"):
        prepare_real_validation(
            tmp_path, {"author": "", "questions": []}, tmp_path / "manifest.json"
        )


def test_pilot_cannot_silently_expand_past_twenty_sources(tmp_path):
    for i in range(21):
        (tmp_path / f"{i}.txt").write_text(f"Unique source {i}")
    with pytest.raises(ValueError, match="twenty"):
        prepare_real_validation(tmp_path, questions(), tmp_path / "manifest.json")
