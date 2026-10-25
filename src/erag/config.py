"""Strict, versioned runtime configuration; unknown keys fail closed."""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    stage: str = "1.1"
    dataset: str = "data/stage-1.1"
    state: str = "artifacts/index"
    max_words: int = 140
    overlap_words: int = 20
    section_boundaries: bool = False
    candidates: int = 20
    rrf_k: int = 60
    bm25_weight: float = 1.0
    vector_weight: float = 1.0
    mode: str = "hybrid"
    rerank: bool = False
    embedding_model: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    embedding_revision: str = ""
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_revision: str = ""
    rerank_window_sentences: int = 0
    rerank_fusion: bool = False
    rerank_initial_weight: float = 1.0
    qdrant_url: str | None = None
    threads: int = 2

    def __post_init__(self):
        if self.stage not in ("1.1", "1.2", "1.3"):
            raise ValueError("only approved stages 1.1, 1.2 and 1.3 are supported")
        for name in ("max_words", "overlap_words", "candidates", "rrf_k", "threads"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 40 <= self.max_words <= 200:
            raise ValueError("max_words must be between 40 and 200 to limit model truncation")
        if not 0 <= self.overlap_words < self.max_words:
            raise ValueError("overlap_words must be nonnegative and smaller than max_words")
        if min(self.candidates, self.rrf_k, self.threads) < 1:
            raise ValueError("candidates, rrf_k and threads must be positive")
        if not isinstance(self.mode, str) or self.mode not in {"bm25", "vector", "hybrid"}:
            raise ValueError("unknown retrieval mode")
        for name in ("embedding_revision", "reranker_revision"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(
                r"[0-9a-f]{40}", getattr(self, name)
            ):
                raise ValueError(f"{name} must be an immutable 40-character commit")
        for name in ("dataset", "state", "embedding_model", "reranker_model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")

        if (
            type(self.rerank_window_sentences) is not int
            or not 0 <= self.rerank_window_sentences <= 3
        ):
            raise ValueError("rerank_window_sentences must be an integer from 0 through 3")
        if type(self.rerank_fusion) is not bool:
            raise ValueError("rerank_fusion must be boolean")
        if self.rerank_fusion and self.rerank_window_sentences == 0:
            raise ValueError("rerank_fusion requires evidence windows")
        if (
            type(self.rerank_initial_weight) not in (int, float)
            or not math.isfinite(self.rerank_initial_weight)
            or self.rerank_initial_weight < 0
        ):
            raise ValueError("rerank_initial_weight must be finite and nonnegative")
        if type(self.rerank) is not bool:
            raise ValueError("rerank must be boolean")
        if type(self.section_boundaries) is not bool:
            raise ValueError("section_boundaries must be boolean")
        for name in ("bm25_weight", "vector_weight"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.qdrant_url is not None and (
            not isinstance(self.qdrant_url, str)
            or not self.qdrant_url.startswith(("http://", "https://"))
        ):
            raise ValueError("qdrant_url must be an HTTP(S) URL or null")

    @property
    def fingerprint(self) -> str:
        # Location, candidate count and ranking knobs do not change index contents.
        keys = ("max_words", "overlap_words", "embedding_model", "embedding_revision")
        values = {k: asdict(self)[k] for k in keys}
        if self.section_boundaries:
            values["section_boundaries"] = True
        if self.stage != "1.1":
            values["stage"] = self.stage
        return digest(values)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("configuration must be a mapping")  # noqa: TRY004
    unknown = set(data) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    return Config(**data)
