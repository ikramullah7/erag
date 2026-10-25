"""The label audit must fail closed for fabricated evidence or changed originals."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "audit_verifier", Path(__file__).resolve().parents[1] / "scripts/validate_label_audit.py"
)
audit_verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_verifier)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def audit_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "audit"
    source = tmp_path / "data/stage-1.1/unit.txt"
    source.parent.mkdir(parents=True)
    source.write_text("The certificate lives in namespace payments.")
    labels = source.with_name("labels.json")
    labels.write_text("[]")
    write_json(
        root / "inputs/questions.json",
        [
            {"question_id": "Q1", "query": "Which namespace?"},
            {"question_id": "Q2", "query": "What is the backup password?"},
        ],
    )
    write_json(
        root / "inputs/units.json",
        [{"unit_id": "C1", "document_id": "unit", "text": source.read_text()}],
    )
    hashes = {name: sha(root / "inputs" / name) for name in ("questions.json", "units.json")}
    write_json(root / "inputs/manifest.json", {"input_sha256": hashes})
    write_json(
        root / "audit-provenance.json",
        {
            "source_sha256": {"unit.txt": sha(source)},
            "original_label_sha256": {"labels.json": sha(labels)},
        },
    )
    rows = [
        {
            "question_id": "Q1",
            "answerability": "answerable",
            "expected_facts": ["payments"],
            "positives": [
                {
                    "unit_id": "C1",
                    "grade": 3,
                    "coverage": "full",
                    "evidence": [source.read_text()],
                    "rationale": "Names the namespace.",
                }
            ],
            "ambiguity": None,
            "requires_human_review": False,
        },
        {
            "question_id": "Q2",
            "answerability": "unanswerable",
            "expected_facts": [],
            "positives": [],
            "ambiguity": None,
            "requires_human_review": True,
        },
    ]
    audit = {"input_hashes": hashes, "coverage_question_count": 2, "judgments": rows}
    write_json(root / "review/independent-judgments.json", audit)
    return root, audit, source


def test_complete_grounded_audit_validates(audit_input):
    root, _, _ = audit_input
    result = audit_verifier.verify_audit(root)
    assert result["validated"] and result["originals_unchanged"]
    assert result["questions"] == 2
    assert result["positive_judgments"] == 1


@pytest.mark.parametrize(
    "error",
    ["fabricated_quote", "duplicate_question", "wrong_input_hash", "grade_coverage_conflict"],
)
def test_invalid_audit_is_rejected(audit_input, error):
    root, audit, _ = audit_input
    if error == "fabricated_quote":
        audit["judgments"][0]["positives"][0]["evidence"] = ["An unsupported answer."]
    elif error == "duplicate_question":
        audit["judgments"].append(audit["judgments"][0])
    elif error == "wrong_input_hash":
        audit["input_hashes"] = {}
    else:
        audit["judgments"][0]["positives"][0]["coverage"] = "partial"
    write_json(root / "review/independent-judgments.json", audit)
    with pytest.raises(ValueError):
        audit_verifier.verify_audit(root)


def test_changed_original_source_is_rejected(audit_input):
    root, _, source = audit_input
    source.write_text("Altered source")
    with pytest.raises(ValueError, match="original source changed"):
        audit_verifier.verify_audit(root)
