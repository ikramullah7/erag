"""Regressions for lost definitions, metadata isolation and untrusted selections."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from erag.answer_config import AnswerConfig
from erag.answering import Generation, answer_question
from erag.context import expand_context, mark_code_fences, retrieval_text
from erag.evidence_answer import evidence_catalog
from erag.pipeline import Engine, build_index


def test_contextual_index_is_versioned_and_raw_offsets_survive(indexed):
    cfg, store, embedder, old = indexed
    changed = replace(cfg, contextual_retrieval=True)
    with pytest.raises(ValueError, match="stale"):
        Engine(changed, store, embedder)
    new = build_index(changed, embedder, store)
    assert new["chunks"] == old["chunks"]
    engine = Engine(changed, store, embedder)
    doc = next(d for d in new["documents"] if d["id"] == engine.chunks[0].document_id)
    assert engine.chunks[0].text == doc["text"][engine.chunks[0].start : engine.chunks[0].end]
    assert retrieval_text(engine.chunks[0], doc["text"]).endswith(engine.chunks[0].text)


def test_neighbor_context_preserves_definitions_and_exact_source_offsets():
    text = "ClusterIP exposes an internal IP.\n\nNodePort exposes a node port.\n\nLoadBalancer exposes an external balancer."
    pieces = text.split("\n\n")
    chunks = []
    for i, piece in enumerate(pieces):
        start = text.index(piece)
        chunks.append(
            {
                "id": str(i),
                "document_id": "d",
                "version": "v",
                "source": "official",
                "position": i,
                "start": start,
                "end": start + len(piece),
                "text": piece,
                "heading": "Types",
            }
        )
    snapshot = {"documents": [{"id": "d", "version": "v", "text": text}], "chunks": chunks}
    result = expand_context(
        {"results": [{"chunk": chunks[1], "final_rank": 1}, {"chunk": chunks[0], "final_rank": 8}]},
        snapshot,
        1,
        140,
    )
    assert len(result["results"]) == 1
    c = result["results"][0]["chunk"]
    assert c["text"] == text[c["start"] : c["end"]] == text
    assert c["text"].count("NodePort") == 1


def selection_engine():
    text = "For each configured volume template, each Pod receives one storage claim."
    chunk = {
        "id": "c",
        "document_id": "d",
        "version": "v",
        "source": "official",
        "position": 0,
        "start": 0,
        "end": len(text),
        "text": text,
        "heading": "Storage",
    }

    class EngineStub:
        cfg = SimpleNamespace(rerank=True, candidates=20)

        def search(self, *args, **kwargs):
            return {
                "results": [{"chunk": chunk, "final_rank": 1}],
                "mode": "hybrid",
                "reranked": True,
                "timings": {},
            }

    return EngineStub(), chunk


class Selector:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, messages, schema):
        return Generation(self.payload, "test", {}, 1)


def test_selection_preserves_condition_and_programmatic_quote_offsets():
    engine, chunk = selection_engine()
    cfg = AnswerConfig(model="local", model_digest="a" * 64, evidence_selection=True)
    result = answer_question(
        engine,
        Selector({"evidence_ids": ["E1"]}),
        cfg,
        "Does every pod get storage?",
    )
    assert result["claims"][0]["text"] == chunk["text"]
    cite = result["claims"][0]["citations"][0]
    assert chunk["text"][cite["start"] : cite["end"]] == cite["quote"]
    assert result["answer_mode"] == "extractive_source_facts"


@pytest.mark.parametrize(
    "payload",
    [
        {"evidence_ids": ["E999"]},
        {"evidence_ids": ["E1", "E1"]},
        {"evidence_ids": "E1"},
        {"status": "insufficient_evidence", "evidence_ids": ["E1"]},
        {
            "status": "answered",
            "evidence_ids": ["E1"],
            "text": "Every Pod gets storage unconditionally.",
        },
    ],
)
def test_untrusted_selector_cannot_invent_rewrite_or_smuggle_answer(payload):
    engine, _ = selection_engine()
    cfg = AnswerConfig(model="local", model_digest="a" * 64, evidence_selection=True)
    result = answer_question(engine, Selector(payload), cfg, "Does every pod get storage?")
    assert result["status"] == "invalid_response" and result["claims"] == []
    assert "unconditionally" not in result["answer"]


def test_catalog_excludes_code_and_questions_but_retains_conditioned_prose():
    sources = [
        {
            "source_id": "S1",
            "chunk": {
                "heading": "Storage",
                "text": "## Storage\n\nDo you need storage?\n\n```\npassword: example-not-a-real-value\n```\n\nFor each configured template, a Pod receives one claim.",
            },
        }
    ]
    facts = evidence_catalog(sources)
    assert len(facts) == 1 and facts[0]["quote"].startswith("For each configured")


def test_long_table_retains_noeviction_rows_with_exact_provenance():
    filler = "\n".join(
        f"| key-{i} | NoSchedule | Pods may run on these nodes with this toleration. |"
        for i in range(30)
    )
    row = "| node-unavailable | NoExecute | These Pods are not evicted while their node is unreachable. |"
    text = filler + "\n" + row
    catalog = evidence_catalog(
        [{"source_id": "S1", "chunk": {"heading": "Tolerations", "text": text}}]
    )
    matching = [c for c in catalog if "not evicted" in c["quote"]]
    assert len(matching) == 1 and row in matching[0]["quote"]
    assert all(c["quote"] in text for c in catalog)


def test_empty_selection_abstains_without_factual_claims():
    engine, _ = selection_engine()
    cfg = AnswerConfig(model="local", model_digest="a" * 64, evidence_selection=True)
    result = answer_question(
        engine, Selector({"evidence_ids": []}), cfg, "What is my actual live address?"
    )
    assert result["status"] == "insufficient_evidence" and result["claims"] == []


def test_wider_balanced_candidates_include_vector_only_results(indexed, monkeypatch):
    cfg, store, embedder, _ = indexed
    engine = Engine(
        replace(cfg, candidates=8, candidate_depth=12, bm25_weight=1, vector_weight=1),
        store,
        embedder,
    )
    lexical = [(c.id, float(12 - i)) for i, c in enumerate(engine.chunks[:12])]
    semantic = [(c.id, float(12 - i)) for i, c in enumerate(engine.chunks[12:24])]
    monkeypatch.setattr(engine.bm25, "search", lambda query, limit: lexical[:limit])
    monkeypatch.setattr(store, "search", lambda collection, vector, limit: semantic[:limit])
    result = engine.search("private endpoint", "hybrid")
    final = {cid for cid, _ in result["initial_ranking"]}
    assert final & {cid for cid, _ in lexical}
    assert final & {cid for cid, _ in semantic}
    assert len(final) == 8


class ReviewSequence:
    def __init__(self, values, changed_model=False):
        self.values, self.changed_model, self.calls = values, changed_model, 0

    def generate(self, messages, schema):
        value = self.values[self.calls]
        self.calls += 1
        return Generation(
            value, "changed" if self.changed_model and self.calls > 1 else "test", {}, 1
        )


def coverage_settings():
    return AnswerConfig(
        model="local", model_digest="a" * 64, evidence_selection=True, coverage_check=True
    )


def test_coverage_review_adds_omitted_alternative_with_original_quotes():
    engine, chunk = selection_engine()
    text = "## Default\n\nOperations run sequentially by default.\n\n## Alternative\n\nThe concurrent policy permits simultaneous operations."
    chunk.update(text=text, end=len(text))
    generator = ReviewSequence([{"evidence_ids": ["E1"]}, {"evidence_ids": ["E1", "E2"]}])
    result = answer_question(
        engine, generator, coverage_settings(), "Can operations run simultaneously?"
    )
    assert generator.calls == 2 and result["status"] == "answered"
    assert "sequentially" in result["answer"] and "simultaneous" in result["answer"]
    assert result["generation_ms"] == 2
    assert result["initial_selection"] == {"evidence_ids": ["E1"]}
    assert all(c["text"] == c["citations"][0]["quote"] for c in result["claims"])


def test_invalid_primary_selection_is_rejected_before_coverage_call():
    engine, _ = selection_engine()
    generator = ReviewSequence([{"evidence_ids": ["E999"]}])
    result = answer_question(engine, generator, coverage_settings(), "Question?")
    assert generator.calls == 1 and result["status"] == "invalid_response"


def test_invalid_coverage_cannot_fall_back_to_exposing_incomplete_primary():
    engine, _ = selection_engine()
    generator = ReviewSequence([{"evidence_ids": ["E1"]}, {"evidence_ids": ["E999"]}])
    result = answer_question(engine, generator, coverage_settings(), "Question?")
    assert result["status"] == "invalid_response" and not result["claims"]


def test_primary_abstention_does_not_trigger_answer_invention_by_coverage():
    engine, _ = selection_engine()
    generator = ReviewSequence([{"evidence_ids": []}])
    result = answer_question(engine, generator, coverage_settings(), "Unknown actual value?")
    assert generator.calls == 1 and result["status"] == "insufficient_evidence"


def test_model_identity_must_remain_stable_during_coverage():
    engine, _ = selection_engine()
    generator = ReviewSequence(
        [{"evidence_ids": ["E1"]}, {"evidence_ids": ["E1"]}], changed_model=True
    )
    with pytest.raises(ValueError, match="changed"):
        answer_question(engine, generator, coverage_settings(), "Question?")


def test_excerpt_starting_inside_code_does_not_hide_following_prose():
    document = "# Reference\n\n```\nmetadata:\n  name: sample\n\n```\n\n## Routing\n\nA selector-free service can target another namespace."
    start = document.index("metadata:")
    sources = [
        {
            "source_id": "S1",
            "chunk": {
                "id": "c",
                "document_id": "d",
                "start": start,
                "text": document[start:],
                "heading": "Routing",
            },
        }
    ]
    mark_code_fences(sources, {"documents": [{"id": "d", "text": document}]})
    assert sources[0]["chunk"]["starts_in_code"] is True
    facts = evidence_catalog(sources)
    assert len(facts) == 1
    assert facts[0]["quote"] == "A selector-free service can target another namespace."


def test_coverage_no_additions_preserves_valid_primary():
    engine, _ = selection_engine()
    generator = ReviewSequence([{"evidence_ids": ["E1"]}, {"evidence_ids": []}])
    result = answer_question(engine, generator, coverage_settings(), "Question?")
    assert result["status"] == "answered" and len(result["claims"]) == 1
    assert result["coverage_additions"] == {"evidence_ids": []}
