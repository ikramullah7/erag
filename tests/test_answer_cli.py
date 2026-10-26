"""CLI answers expose validated output and preserve failure exit codes."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from erag import cli
from erag.answer_config import load_answer_config
from erag.codex import CodexGenerator
from erag.config import load_config


@pytest.mark.parametrize(
    "status,expected",
    [
        ("answered", 0),
        ("insufficient_evidence", 0),
        ("invalid_response", 2),
    ],
)
def test_answer_cli_uses_answer_profile_and_exposes_status(
    monkeypatch, capsys, tmp_path, status, expected
):
    cfg = replace(load_config("config/default.yaml"), mode="bm25", rerank=False)
    paths = []
    closed = []

    def config(path):
        paths.append(path)
        return cfg

    monkeypatch.setattr(cli, "load_config", config)
    monkeypatch.setattr(
        cli, "QdrantStore", lambda **kw: SimpleNamespace(close=lambda: closed.append(True))
    )
    monkeypatch.setattr(cli, "Engine", lambda *args: object())
    monkeypatch.setattr(cli, "OllamaGenerator", lambda config: object())
    result = {"status": status, "answer": "A validated answer", "claims": [], "sources": []}
    monkeypatch.setattr(cli, "answer_question", lambda *args: result)
    monkeypatch.setattr(sys, "argv", ["erag", "answer", "Question?", "--json"])
    assert cli.main() == expected
    assert json.loads(capsys.readouterr().out)["status"] == status
    assert paths == [load_answer_config("config/answer.yaml").retrieval_config]
    assert closed == [True]


def test_answer_cli_provider_failure_is_nonzero_and_closes_store(monkeypatch, capsys):
    cfg = replace(load_config("config/default.yaml"), mode="bm25", rerank=False)
    closed = []
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        cli, "QdrantStore", lambda **kw: SimpleNamespace(close=lambda: closed.append(True))
    )
    monkeypatch.setattr(cli, "Engine", lambda *args: object())
    monkeypatch.setattr(cli, "OllamaGenerator", lambda config: object())

    def fail(*args):
        raise RuntimeError("model digest changed")

    monkeypatch.setattr(cli, "answer_question", fail)
    monkeypatch.setattr(sys, "argv", ["erag", "answer", "Question?"])
    assert cli.main() == 1
    assert capsys.readouterr().out == "" and closed == [True]


def test_answer_cli_routes_codex_flag_through_shared_answer_pipeline(monkeypatch, capsys):
    cfg = replace(load_config("config/default.yaml"), mode="bm25", rerank=False)
    answer_cfg = replace(load_answer_config("config/answer.yaml"), provider="codex")
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "load_answer_config", lambda path: answer_cfg)
    monkeypatch.setattr(cli, "QdrantStore", lambda **kw: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli, "Engine", lambda *args: object())

    def answer(engine, generator, settings, *args):
        assert isinstance(generator, CodexGenerator)
        assert settings.provider == "codex"
        return {
            "status": "answered",
            "answer": "Validated source fact",
            "claims": [],
            "sources": [],
        }

    monkeypatch.setattr(cli, "answer_question", answer)
    monkeypatch.setattr(sys, "argv", ["erag", "answer", "Question?"])
    assert cli.main() == 0
    assert "Validated source fact" in capsys.readouterr().out


def test_semantic_cli_blocks_scoring_when_judge_calibration_fails(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "OllamaGenerator", lambda config: object())
    monkeypatch.setattr(cli, "calibrate_judge", lambda *args: {"passed": False})
    monkeypatch.setattr(
        cli,
        "abstention_only",
        lambda *args: {
            "summary": {
                "faithfulness": None,
                "answer_correctness": None,
                "citation_accuracy": None,
                "abstention_accuracy": 1,
                "judge_errors": [],
            }
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("uncalibrated judge must not score answers")

    monkeypatch.setattr(cli, "evaluate_semantics", forbidden)
    monkeypatch.setattr(cli, "QdrantStore", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "erag",
            "score-answers",
            "--answers",
            "saved.json",
            "--output",
            str(tmp_path / "scores.json"),
        ],
    )
    assert cli.main() == 2
    assert "calibration failed" in capsys.readouterr().out


def test_semantic_cli_scores_saved_answers_without_loading_retrieval(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "OllamaGenerator", lambda config: object())
    monkeypatch.setattr(cli, "calibrate_judge", lambda *args: {"passed": True})
    report = {
        "summary": {
            "faithfulness": 0.75,
            "answer_correctness": 0.5,
            "citation_accuracy": 0.5,
            "abstention_accuracy": 1,
            "judge_errors": [],
            "semantic_scoring_complete": True,
        }
    }
    monkeypatch.setattr(cli, "evaluate_semantics", lambda *args: report)

    def forbidden(*args, **kwargs):
        raise AssertionError("offline scoring does not need an active retrieval index")

    monkeypatch.setattr(cli, "QdrantStore", forbidden)
    path = tmp_path / "scores.json"
    monkeypatch.setattr(
        sys, "argv", ["erag", "score-answers", "--answers", "saved.json", "--output", str(path)]
    )
    assert cli.main() == 0
    assert "75.00%" in capsys.readouterr().out
    assert json.loads(path.read_text())["calibration"]["passed"]
