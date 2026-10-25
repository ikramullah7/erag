"""CPU-only model adapters. Loading is lazy; BM25 needs no model downloads."""

import re

import numpy as np

from .config import Config
from .fusion import rrf


def configure_torch(threads: int) -> None:
    import torch

    torch.manual_seed(0)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


class SentenceEmbedder:
    def __init__(self, cfg: Config):
        from sentence_transformers import SentenceTransformer

        configure_torch(cfg.threads)
        self.identity = f"{cfg.embedding_model}@{cfg.embedding_revision}"
        self.model = SentenceTransformer(
            cfg.embedding_model,
            revision=cfg.embedding_revision,
            device="cpu",
            trust_remote_code=False,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        # Fail rather than silently truncate indexed evidence.
        lengths = [len(self.model.tokenizer.encode(t)) for t in texts]
        if max(lengths, default=0) > self.model.max_seq_length:
            raise ValueError("embedding input exceeds model token budget; reduce chunk size")
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float32)


def sentence_windows(text: str, size: int) -> list[str]:
    if size == 0:
        return [text]
    if size < 1:
        raise ValueError("window sentence count must be nonnegative")
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    if len(sentences) <= size:
        return [text]
    return [" ".join(sentences[i : i + size]) for i in range(len(sentences) - size + 1)]


class CrossEncoderReranker:
    def __init__(self, cfg: Config):
        from sentence_transformers import CrossEncoder
        from torch import nn

        configure_torch(cfg.threads)
        self.window_sentences = cfg.rerank_window_sentences
        self.rank_fusion = cfg.rerank_fusion
        self.initial_weight = cfg.rerank_initial_weight
        self.rrf_k = cfg.rrf_k
        self.score_type = "reranker_rrf" if self.rank_fusion else "cross_encoder"
        self.identity = (
            f"{cfg.reranker_model}@{cfg.reranker_revision}:windows={self.window_sentences}"
        )
        self.model = CrossEncoder(
            cfg.reranker_model,
            activation_fn=nn.Identity(),
            revision=cfg.reranker_revision,
            device="cpu",
            trust_remote_code=False,
        )

    def score(self, query: str, texts: list[str]) -> list[float]:
        groups = [sentence_windows(text, self.window_sentences) for text in texts]
        pairs = [(query, window) for windows in groups for window in windows]
        if not pairs:
            return []
        if any(
            len(self.model.tokenizer.encode(q, t)) > self.model.max_seq_length for q, t in pairs
        ):
            raise ValueError("reranker input exceeds token budget")
        scores = np.asarray(self.model.predict(pairs, show_progress_bar=False)).reshape(-1).tolist()
        if len(scores) != len(pairs) or not np.isfinite(scores).all():
            raise ValueError("invalid cross-encoder window scores")
        result, start = [], 0
        for windows in groups:
            result.append(max(scores[start : start + len(windows)]))
            start += len(windows)
        if getattr(self, "rank_fusion", False):
            context_pairs = [(query, text) for text in texts]
            if any(
                len(self.model.tokenizer.encode(q, t)) > self.model.max_seq_length
                for q, t in context_pairs
            ):
                raise ValueError("context reranker input exceeds token budget")
            context_scores = (
                np.asarray(self.model.predict(context_pairs, show_progress_bar=False))
                .reshape(-1)
                .tolist()
            )
            if len(context_scores) != len(texts) or not np.isfinite(context_scores).all():
                raise ValueError("invalid context reranker scores")
            views = [
                sorted(
                    [(str(i), score) for i, score in enumerate(values)],
                    key=lambda x: (-x[1], int(x[0])),
                )
                for values in (context_scores, result)
            ]
            weights = [1.0, 1.0]
            if self.initial_weight > 0:
                views.append([(str(i), 0.0) for i in range(len(texts))])
                weights.append(self.initial_weight)
            fused = dict(rrf(views, self.rrf_k, weights))
            return [fused[str(i)] for i in range(len(texts))]
        return result
