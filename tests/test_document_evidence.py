"""Distant-rule rescue and failure traces must retain exact evidence provenance."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from erag.answer_config import AnswerConfig
from erag.answering import Generation, answer_question
from erag.document_evidence import search_document_facts
from erag.evidence_answer import (
    SYSTEM,
    coverage_messages,
    evidence_catalog,
    rank_catalog,
    select_answer,
    selection_messages,
)


def test_document_fact_search_recovers_distant_rule_without_searching_other_documents():
    documents, anchors = [], []
    for i in range(3):
        text = f"# Reference {i}\n\nA general introductory statement.\n\n## Readiness\n\nAvailability requires readiness for the configured interval."
        doc = {"id": str(i), "version": "v", "text": text, "source": str(i)}
        chunk = {
            "id": str(i),
            "document_id": str(i),
            "version": "v",
            "source": str(i),
            "heading": "Intro",
            "position": 0,
            "start": 0,
            "end": 15,
            "text": text[:15],
        }
        documents.append(doc)
        anchors.append({"chunk": chunk, "final_rank": i + 1})

    class Ranker:
        def score(self, question, texts):
            assert all("Reference 2" not in t for t in texts)
            return [10 if "requires readiness" in t else 0 for t in texts]

    cfg = AnswerConfig(model="local", model_digest="a" * 64, context_chunks=2)
    retrieval = {"results": anchors, "mode": "hybrid"}
    result = search_document_facts(
        "availability", retrieval, {"documents": documents}, Ranker(), cfg
    )
    assert result["results"][0]["chunk"]["start"] > anchors[0]["chunk"]["end"]
    assert retrieval["results"] == anchors  # Canonical raw ranks are not rewritten.
    for row in result["results"]:
        chunk = row["chunk"]
        doc = documents[int(chunk["document_id"])]
        assert chunk["text"] == doc["text"][chunk["start"] : chunk["end"]]
        assert chunk["document_id"] in {"0", "1"}


def test_section_diversity_keeps_rule_amid_repeated_table_rows():
    catalog = [
        {
            "id": str(i),
            "document": "d",
            "heading": "Table" if i < 6 else "Policy",
            "quote": f"A factual source statement {i}.",
        }
        for i in range(7)
    ]
    ranker = SimpleNamespace(score=lambda q, texts: list(range(7, 0, -1)))
    assert [c["id"] for c in rank_catalog("q", catalog, ranker, 4)] == ["0", "1", "2", "3"]
    assert [c["id"] for c in rank_catalog("q", catalog, ranker, 4, True)] == ["0", "6", "1", "2"]


def test_example_narrative_is_not_exposed_as_an_actual_failure():
    text = "# Example\n\nFor example, an operator has ten replicas.\n\nYou update to an image that cannot be resolved.\n\n## Instructions\n\nYou update the policy by setting the documented field."
    facts = evidence_catalog(
        [{"source_id": "S1", "chunk": {"heading": "Reference", "text": text, "start": 100}}]
    )
    assert not any("cannot be resolved" in f["quote"] for f in facts)
    assert any("documented field" in f["quote"] for f in facts)
    assert all(text[f["start"] - 100 : f["end"] - 100] == f["quote"] for f in facts)


@pytest.mark.parametrize("phase", ["primary", "coverage_review"])
def test_provider_failure_preserves_exact_context_and_catalog(phase):
    text = "A requested replica count is a target, not proof of readiness."
    chunk = {
        "id": "c",
        "document_id": "d",
        "version": "v",
        "source": "official",
        "heading": "Readiness",
        "start": 0,
        "end": len(text),
        "text": text,
    }
    engine = SimpleNamespace(
        cfg=SimpleNamespace(rerank=True, candidates=20),
        search=lambda *a, **k: {
            "results": [{"chunk": chunk, "final_rank": 1}],
            "mode": "hybrid",
            "reranked": True,
            "timings": {},
        },
    )

    class Generator:
        def generate(self, *args):
            if phase == "primary":
                raise RuntimeError("timed out")
            return Generation({"evidence_ids": ["E1"]}, "test", {}, 1)

        def review(self, *args):
            raise RuntimeError("timed out")

    cfg = AnswerConfig(
        model="local", model_digest="a" * 64, evidence_selection=True, coverage_check=True
    )
    result = answer_question(engine, Generator(), cfg, "What does the replica count mean?")
    assert result["status"] == "provider_error" and result["provider_error_phase"] == phase
    assert result["sources"][0]["chunk"] == chunk and result["catalog"][0]["quote"] == text
    assert result["claims"] == [] and result["failed_generation_ms"] >= 0
    assert result["generation_ms"] >= result["failed_generation_ms"]


def test_document_search_requires_explicit_validated_selection_configuration():
    cfg = AnswerConfig(model="local", model_digest="a" * 64)
    with pytest.raises(ValueError, match="requires"):
        replace(cfg, document_fact_search=True)
    with pytest.raises(ValueError, match="boolean"):
        replace(cfg, fact_section_coverage=1)


def test_lexical_coverage_retains_a_definition_missed_by_semantic_ranking():
    catalog = [
        {
            "id": str(i),
            "document": "manual",
            "heading": "Counters",
            "quote": f"The worked example discusses replica totals and counter values in step {i}.",
        }
        for i in range(15)
    ]
    catalog[0]["quote"] = (
        "The desired and available counts in this worked example have different totals."
    )
    catalog[-1].update(
        heading="Availability",
        quote="An instance is considered available only after readiness succeeds for the required interval.",
    )
    ranker = SimpleNamespace(score=lambda q, texts: list(range(len(texts), 0, -1)))
    question = "Topic: Manual\nQuestion: Why desired counts but fewer available?"
    assert "14" not in {c["id"] for c in rank_catalog(question, catalog, ranker, 4)}
    selected = rank_catalog(question, catalog, ranker, 4, True, True)
    assert len(selected) == 4 and len({c["id"] for c in selected}) == 4
    assert "14" in {c["id"] for c in selected}


def test_extra_source_budget_is_bounded_and_extractive_only():
    cfg = AnswerConfig(model="local", model_digest="a" * 64)
    with pytest.raises(ValueError, match="budget"):
        replace(cfg, context_chunks=40)
    assert replace(cfg, evidence_selection=True, context_chunks=40).context_chunks == 40
    with pytest.raises(ValueError, match="budget"):
        replace(cfg, evidence_selection=True, context_chunks=41)


def test_coverage_review_reuses_the_primary_prompt_prefix():
    text = "The configured count is a target, and readiness is checked separately."
    chunk = {
        "id": "c",
        "document_id": "d",
        "version": "v",
        "source": "official",
        "heading": "Readiness",
        "start": 0,
        "end": len(text),
        "text": text,
    }
    engine = SimpleNamespace(
        cfg=SimpleNamespace(rerank=True, candidates=20),
        search=lambda *a, **k: {
            "results": [{"chunk": chunk, "final_rank": 1}],
            "mode": "hybrid",
            "reranked": True,
            "timings": {},
        },
    )

    class Recording:
        def __init__(self):
            self.messages = []

        def generate(self, messages, schema):
            self.messages.append(messages)
            return Generation(
                {"evidence_ids": ["E1"] if len(self.messages) == 1 else []}, "test", {}, 1
            )

    cfg = AnswerConfig(
        model="local", model_digest="a" * 64, evidence_selection=True, coverage_check=True
    )
    generator = Recording()
    result = answer_question(engine, generator, cfg, "What is the configured count?")
    assert result["status"] == "answered"
    primary, review = generator.messages
    assert review[: len(primary)] == primary
    assert review[2]["role"] == "assistant" and review[3]["role"] == "user"
    assert sum(m["content"].count('"catalog":') for m in review) == 1


def test_primary_catalog_reserves_enough_space_for_the_coverage_review():
    text = "\n\n".join(
        f"Rule {i} constrains the configured target and explains its permitted behavior. "
        + "Supporting conditions must be preserved. " * 4
        for i in range(20)
    )
    chunk = {
        "id": "c",
        "document_id": "d",
        "version": "v",
        "source": "official",
        "heading": "Rules",
        "start": 0,
        "end": len(text),
        "text": text,
    }
    engine = SimpleNamespace(
        cfg=SimpleNamespace(rerank=True, candidates=20),
        search=lambda *a, **k: {
            "results": [{"chunk": chunk, "final_rank": 1}],
            "mode": "hybrid",
            "reranked": True,
            "timings": {},
        },
    )
    cfg = AnswerConfig(
        model="local",
        model_digest="a" * 64,
        num_ctx=8192,
        num_predict=64,
        evidence_selection=True,
        coverage_check=True,
    )

    class BudgetChecking:
        calls = 0

        def generate(self, messages, schema):
            size = sum(len(m["content"].encode()) for m in messages) + len(
                json.dumps(schema).encode()
            )
            assert size + cfg.num_predict + 512 <= cfg.num_ctx
            self.calls += 1
            return Generation({"evidence_ids": ["E1"] if self.calls == 1 else []}, "test", {}, 1)

    generator = BudgetChecking()
    result = answer_question(engine, generator, cfg, "What constraints apply?")
    assert result["status"] == "answered" and generator.calls == 2
    assert result["context_budget"]["coverage_review_reserved_bytes"] > 0


def test_document_rescue_is_additive_without_displacing_primary_evidence():
    def source(sid, text):
        return {
            "source_id": sid,
            "chunk": {
                "id": sid,
                "document_id": sid,
                "version": "v",
                "source": "official",
                "heading": "Policy",
                "start": 0,
                "end": len(text),
                "text": text,
            },
        }

    primary = source(
        "S1", "The controller replaces a deleted worker to maintain the desired count."
    )
    rescue = source("D1", "By default a worker must be ready before it is counted as available.")
    cfg = AnswerConfig(
        model="local",
        model_digest="a" * 64,
        evidence_selection=True,
        coverage_check=True,
        document_fact_search=True,
        fact_candidates=8,
    )
    ranker = SimpleNamespace(score=lambda q, texts: [1.0] * len(texts))

    class Recording:
        def __init__(self):
            self.messages = []

        def generate(self, messages, schema):
            self.messages.append(messages)
            return Generation(
                {"evidence_ids": ["E1"] if len(self.messages) == 1 else ["R1"]}, "test", {}, 1
            )

    generator = Recording()
    result = select_answer(
        generator,
        cfg,
        "How does replacement work?",
        [primary],
        {"question": "How does replacement work?"},
        ranker,
        [rescue],
    )
    assert result["status"] == "answered"
    assert [c["text"] for c in result["claims"]] == [
        primary["chunk"]["text"],
        rescue["chunk"]["text"],
    ]
    first, review = generator.messages
    assert review[:2] == first
    assert rescue["chunk"]["text"] not in first[1]["content"]
    assert rescue["chunk"]["text"] in review[-1]["content"]
    assert len(result["primary_catalog"]) == 1 and len(result["catalog"]) == 2


def test_supported_primary_abstention_can_be_rescued_only_in_document_search_mode():
    def source(sid, text):
        return {
            "source_id": sid,
            "chunk": {
                "id": sid,
                "document_id": sid,
                "version": "v",
                "source": "official",
                "heading": "Mapping",
                "start": 0,
                "end": len(text),
                "text": text,
            },
        }

    primary = source("S1", "An overview describes the application platform and its components.")
    rescue = source(
        "D1", "The binding maps an incoming port to the configured application destination port."
    )
    cfg = AnswerConfig(
        model="local",
        model_digest="a" * 64,
        evidence_selection=True,
        coverage_check=True,
        document_fact_search=True,
        fact_candidates=8,
    )
    ranker = SimpleNamespace(score=lambda q, texts: [1.0] * len(texts))

    class Sequence:
        calls = 0

        def generate(self, messages, schema):
            self.calls += 1
            return Generation({"evidence_ids": [] if self.calls == 1 else ["R1"]}, "test", {}, 1)

    generator = Sequence()
    result = select_answer(
        generator,
        cfg,
        "What happens with a destination mismatch?",
        [primary],
        {"question": "What happens with a destination mismatch?"},
        ranker,
        [rescue],
    )
    assert result["status"] == "answered" and result["abstention_rescued"]
    assert result["claims"][0]["text"] == rescue["chunk"]["text"]
    assert result["initial_status"] == "insufficient_evidence" and generator.calls == 2


def test_completion_preserves_original_fact_content_and_uses_exact_same_version():
    full = "A worker has a stable identity. That identity survives rescheduling."
    fragment = full[2:]
    primary = {
        "source_id": "S1",
        "chunk": {
            "id": "c",
            "document_id": "d",
            "version": "v",
            "source": "official",
            "heading": "Identity",
            "start": 2,
            "end": len(full),
            "text": fragment,
        },
    }
    rescue = {
        "source_id": "D1",
        "chunk": {**primary["chunk"], "id": "full", "start": 0, "text": full},
    }
    cfg = AnswerConfig(
        model="local",
        model_digest="a" * 64,
        evidence_selection=True,
        coverage_check=True,
        document_fact_search=True,
        fact_candidates=8,
    )
    ranker = SimpleNamespace(score=lambda q, texts: [1.0] * len(texts))

    class Select:
        calls = 0

        def generate(self, messages, schema):
            self.calls += 1
            return Generation({"evidence_ids": ["E1"] if self.calls == 1 else []}, "test", {}, 1)

    result = select_answer(
        Select(),
        cfg,
        "Is identity stable?",
        [primary],
        {"question": "Is identity stable?"},
        ranker,
        [rescue],
    )
    assert result["claims"][0]["text"] == full
    assert fragment in result["claims"][0]["text"]
    rescue["chunk"]["version"] = "other"
    result = select_answer(
        Select(),
        cfg,
        "Is identity stable?",
        [primary],
        {"question": "Is identity stable?"},
        ranker,
        [rescue],
    )
    assert result["claims"][0]["text"] == fragment


def test_recovery_uses_a_fresh_question_focused_selection_contract():
    primary = [{"id": "E1", "quote": "Original source fact."}]
    extra = [{"id": "R1", "quote": "New mapping rule."}]
    messages = coverage_messages(
        "What happens with a mapping mismatch?", primary + extra, {"evidence_ids": []}, primary
    )
    assert messages[0] == {"role": "system", "content": SYSTEM}
    assert len(messages) == 2
    data = json.loads(messages[-1]["content"])
    assert data["catalog"] == primary + extra
    assert list(data)[-1] == "question"


def test_additive_review_repeats_question_after_new_facts_and_keeps_cache_prefix():
    primary = [{"id": "E1", "quote": "Original source fact."}]
    extra = [{"id": "R1", "quote": "An additional qualifier."}]
    question = "Which conditions apply?"
    messages = coverage_messages(question, primary + extra, {"evidence_ids": ["E1"]}, primary)
    assert messages[:2] == selection_messages(question, primary)
    tail = json.loads(messages[-1]["content"].rsplit("\n", 1)[1])
    assert tail["additional_catalog"] == extra and tail["question"] == question
    assert list(tail)[-1] == "question"
