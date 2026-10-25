import math

import pytest

from erag.bm25 import tokenize
from erag.evaluation import quality_gate, ranking_metrics
from erag.fusion import rrf


def test_metrics_known_values_and_missing_relevant_items():
    m = ranking_metrics(["x", "b", "a"], {"a": 3, "b": 1, "c": 1}, [1, 3, 5])
    assert m["mrr"] == 0.5
    assert m["recall@1"] == 0
    assert m["recall@3"] == pytest.approx(2 / 3)
    expected = (1 / math.log2(3) + 7 / math.log2(4)) / (7 + 1 / math.log2(3) + 1 / math.log2(4))
    assert m["ndcg@3"] == pytest.approx(expected)


def test_perfect_metrics():
    assert all(v == 1 for v in ranking_metrics(["a", "b"], {"a": 3, "b": 2}, [5]).values())


def test_no_answer_not_misleading_zero():
    assert ranking_metrics(["a"], {}, [3, 5]) == {}


def test_duplicate_rank_rejected():
    with pytest.raises(ValueError):
        ranking_metrics(["a", "a"], {"a": 1}, [3])


def test_rrf_uses_rank_not_score_and_is_deterministic():
    a = rrf([[("a", 100), ("b", 2)], [("b", 0.9), ("a", 0.1)]])
    b = rrf([[("a", 1), ("b", -100)], [("b", 10000), ("a", 9999)]])
    assert a == b
    assert a[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_duplicate_rejected():
    with pytest.raises(ValueError):
        rrf([[("a", 1), ("a", 0.5)]])


def test_technical_identifier_tokens():
    tokens = tokenize("payments-api-tls TCP 443 10.42.8.17")
    assert "payments-api-tls" in tokens and "tls" in tokens and "10.42.8.17" in tokens


def test_quality_gate_does_not_hide_failure():
    summary = {
        "document": {"recall@3": 1, "mrr": 1, "ndcg@5": 1},
        "chunk": {"recall@5": 1, "ndcg@10": 0.79},
        "latency_ms_p95": 10,
    }
    assert not quality_gate(summary, False)["passed"]


def test_hit_measures_any_relevant_chunk_not_all_relevant_chunks():
    metrics = ranking_metrics(["wrong", "a", "other"], {"a": 3, "b": 1}, [3, 5, 10])
    assert metrics["hit@1"] == 0
    assert metrics["hit@3"] == 1
    assert metrics["recall@5"] == 0.5


def test_empty_results_have_zero_hits_for_answerable_query():
    metrics = ranking_metrics([], {"a": 3}, [5, 10])
    assert metrics["hit@1"] == metrics["hit@3"] == metrics["mrr"] == 0


@pytest.mark.parametrize(
    "metric,threshold",
    [("hit@1", 0.95), ("hit@3", 0.98), ("recall@5", 0.95), ("mrr", 0.95), ("ndcg@10", 0.95)],
)
def test_each_new_target_independently_blocks_stage_acceptance(metric, threshold):
    summary = {
        "document": {"recall@3": 1, "mrr": 1, "ndcg@5": 1},
        "chunk": {"hit@1": 1, "hit@3": 1, "recall@5": 1, "mrr": 1, "ndcg@10": 1},
        "latency_ms_p95": 10,
    }
    summary["chunk"][metric] = threshold - 0.000001
    assert not quality_gate(summary, False)["passed"]
    summary["chunk"][metric] = threshold
    assert quality_gate(summary, False)["passed"]


def test_old_gate_cannot_pass_without_new_hit_metrics():
    summary = {
        "document": {"recall@3": 1, "mrr": 1, "ndcg@5": 1},
        "chunk": {"recall@5": 1, "mrr": 1, "ndcg@10": 1},
        "latency_ms_p95": 10,
    }
    assert not quality_gate(summary, False)["passed"]


def test_weighted_rrf_prefers_stronger_rank_provider_without_using_raw_scores():
    rankings = [[("a", 1), ("b", 0.99)], [("b", 1000), ("a", -1000)]]
    result = rrf(rankings, 60, [3, 1])
    assert result[0][0] == "a"
    assert result[0][1] == pytest.approx(3 / 61 + 1 / 62)
    assert rrf(rankings, 60, [1, 1]) == rrf(rankings, 60)


@pytest.mark.parametrize("weights", [[], [1], [1, 0], [1, -1], [1, float("nan")], [1, True]])
def test_invalid_rrf_weights_rejected(weights):
    with pytest.raises(ValueError):
        rrf([[("a", 1)], [("b", 2)]], 60, weights)


def test_sentence_windows_keep_technical_identifiers_and_neighbouring_context():
    from erag.models import sentence_windows

    text = "Endpoint 10.42.8.17 serves ledger.internal.example.test. Its certificate is valid. Check the route."
    windows = sentence_windows(text, 2)
    assert len(windows) == 2
    assert "10.42.8.17" in windows[0] and "ledger.internal.example.test" in windows[0]
    assert all("Its certificate is valid." in w for w in windows)
    assert sentence_windows(text, 0) == [text]


def test_window_reranker_returns_best_evidence_score_for_each_original_chunk():
    import numpy as np

    from erag.models import CrossEncoderReranker

    class Tokenizer:
        def encode(self, query, text):
            return [1, 2]

    class Model:
        tokenizer = Tokenizer()
        max_seq_length = 512

        def predict(self, pairs, show_progress_bar=False):
            assert len(pairs) == 3
            return np.array([2.0, 5.0, 3.0])

    reranker = object.__new__(CrossEncoderReranker)
    reranker.window_sentences = 2
    reranker.model = Model()
    assert reranker.score(
        "query", ["First fact. Second fact. Third fact.", "Another passage."]
    ) == [5.0, 3.0]


@pytest.mark.parametrize("scores", [[1.0], [1.0, float("nan")]])
def test_window_reranker_rejects_invalid_prediction_arrays(scores):
    import numpy as np

    from erag.models import CrossEncoderReranker

    class Tokenizer:
        def encode(self, query, text):
            return [1, 2]

    class Model:
        tokenizer = Tokenizer()
        max_seq_length = 512

        def predict(self, pairs, show_progress_bar=False):
            return np.array(scores)

    reranker = object.__new__(CrossEncoderReranker)
    reranker.window_sentences = 0
    reranker.model = Model()
    with pytest.raises(ValueError, match="invalid cross-encoder"):
        reranker.score("query", ["First passage.", "Second passage."])


def test_reranker_context_window_and_initial_views_are_fused_by_rank_only():
    import numpy as np

    from erag.models import CrossEncoderReranker

    class Tokenizer:
        def encode(self, query, text):
            return [1, 2]

    class Model:
        tokenizer = Tokenizer()
        max_seq_length = 512
        calls = 0

        def predict(self, pairs, show_progress_bar=False):
            self.calls += 1
            return np.array([10.0, 0.0]) if self.calls == 1 else np.array([-100.0, 100.0])

    reranker = object.__new__(CrossEncoderReranker)
    reranker.window_sentences = 2
    reranker.rank_fusion = True
    reranker.initial_weight = 1
    reranker.rrf_k = 60
    reranker.model = Model()
    scores = reranker.score("query", ["First passage.", "Second passage."])
    assert scores[0] == pytest.approx(1 / 62 + 2 / 61)
    assert scores[1] == pytest.approx(1 / 61 + 2 / 62)
    assert scores[0] > scores[1]
