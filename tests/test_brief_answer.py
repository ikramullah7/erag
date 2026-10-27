"""Brief answers must remain supported, complete and genuinely short."""

from copy import deepcopy
from dataclasses import replace

import pytest

from erag.answer_config import AnswerConfig
from erag.answering import Generation
from erag.brief_answer import synthesize_brief_answer


@pytest.fixture
def sample():
    quote = "Internal mode exposes the service within the cluster. Node mode also exposes a port on each node."
    chunk = {
        "id": "c",
        "document_id": "d",
        "version": "v",
        "source": "manual",
        "heading": "Access",
        "text": quote,
        "start": 0,
        "end": len(quote),
    }
    claim = {"text": quote, "citations": [{"source_id": "S1", "quote": quote}]}
    result = {
        "question": "How do Internal and Node modes differ?",
        "status": "answered",
        "answer": quote,
        "claims": [claim],
        "sources": [{"source_id": "S1", "chunk": chunk}],
        "model_identity": "test",
        "generation_ms": 1,
    }
    cfg = AnswerConfig(
        model="local", model_digest="a" * 64, evidence_selection=True, brief_answers=True
    )
    payload = {
        "status": "answered",
        "claims": [
            {
                "text": "Internal mode provides internal access; Node mode additionally exposes a port on each node.",
                "citations": [{"source_id": "S1", "quote": quote}],
            }
        ],
    }
    return result, cfg, payload


class Generator:
    def __init__(self, payload, verdict=None, identity="test"):
        self.payload = payload
        self.verdict = (
            {"claims_supported": [True], "complete": True, "focused": True}
            if verdict is None
            else verdict
        )
        self.identity = identity
        self.calls = []

    def generate(self, messages, schema):
        self.calls.append((messages, schema))
        return Generation(self.payload, self.identity, {}, 2)

    def review(self, messages, schema):
        self.calls.append((messages, schema))
        return Generation(self.verdict, self.identity, {}, 3)


def test_synthesis_is_short_cited_and_preserves_inspectable_evidence(sample):
    result, cfg, payload = sample
    original = deepcopy(result)
    gen = Generator(payload)
    out = synthesize_brief_answer(result, gen, cfg)
    assert result == original
    assert out["status"] == "answered" and out["brief_word_count"] < 30
    assert out["answer"] != result["answer"] and "[S1]" in out["answer"]
    assert (
        out["evidence_claims"] == original["claims"]
        and out["evidence_answer"] == original["answer"]
    )
    assert out["claims"][0]["citations"][0]["start"] == 0
    assert len(gen.calls) == 2 and out["generation_ms"] == 6


@pytest.mark.parametrize(
    "verdict",
    [
        {"claims_supported": [False], "complete": True, "focused": True},
        {"claims_supported": [True], "complete": False, "focused": True},
        {"claims_supported": [True], "complete": True, "focused": False},
        {"claims_supported": ["true"], "complete": True, "focused": True},
    ],
)
def test_unsupported_missing_qualifier_repetitive_or_malformed_answers_fail_closed(sample, verdict):
    result, cfg, payload = sample
    out = synthesize_brief_answer(result, Generator(payload, verdict), cfg)
    assert out["status"] == "invalid_response" and out["claims"] == []
    assert out["evidence_claims"] == result["claims"]
    assert out["answer"] != result["answer"]  # Never silently fall back to the long answer.


@pytest.mark.parametrize("problem", ["length", "quote", "model"])
def test_oversized_invented_quotes_and_changed_model_are_rejected(sample, problem):
    result, cfg, payload = sample
    if problem == "length":
        payload["claims"][0]["text"] = "internal " * 101
    if problem == "quote":
        payload["claims"][0]["citations"][0]["quote"] = "A fabricated source quotation."
    gen = Generator(payload, identity="changed" if problem == "model" else "test")
    out = synthesize_brief_answer(result, gen, cfg)
    assert out["status"] == "invalid_response" and len(gen.calls) == 1


def test_abstentions_and_provider_failures_are_not_rewritten(sample):
    result, cfg, payload = sample
    for status in ("provider_error", "insufficient_evidence", "invalid_response"):
        failed = {**result, "status": status, "claims": []}
        gen = Generator(payload)
        assert synthesize_brief_answer(failed, gen, cfg) == failed and not gen.calls


def test_brief_configuration_is_explicit_and_bounded(sample):
    _, cfg, _ = sample
    for values in ({"brief_answers": 1}, {"brief_max_words": 500}, {"evidence_selection": False}):
        with pytest.raises(ValueError):
            replace(cfg, **values)


def test_one_revision_can_restore_missing_qualifier_and_is_reviewed_again(sample):
    result, cfg, payload = sample

    class Revising(Generator):
        def review(self, messages, schema):
            verdict = {"claims_supported": [True], "complete": len(self.calls) > 2, "focused": True}
            self.calls.append((messages, schema))
            return Generation(verdict, "test", {}, 3)

    gen = Revising(payload)
    out = synthesize_brief_answer(result, gen, cfg)
    assert out["status"] == "answered" and len(gen.calls) == 4
    assert "revision_required" in gen.calls[2][0][1]["content"]
    assert [a["accepted"] for a in out["brief_attempts"]] == [False, True]
    assert out["generation_ms"] == 11


def test_review_timeout_is_reported_without_falling_back_to_long_answer(sample):
    result, cfg, payload = sample

    class Timeout(Generator):
        def review(self, *args):
            raise RuntimeError("review timed out")

    out = synthesize_brief_answer(result, Timeout(payload), cfg)
    assert out["status"] == "provider_error" and out["claims"] == []
    assert out["brief_error_phase"] == "brief_review"


def test_two_quotes_from_same_source_render_one_compact_marker(sample):
    result, cfg, payload = sample
    payload["claims"][0]["citations"] *= 2
    out = synthesize_brief_answer(result, Generator(payload), cfg)
    assert out["status"] == "answered" and out["answer"].count("[S1]") == 1
    assert len(out["claims"][0]["citations"]) == 2
