"""Immutable source records and replaceable retrieval provider contracts."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Document:
    id: str
    source: str
    version: str
    text: str
    dataset: str = "stage-1.1-v1"


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    source: str
    version: str
    position: int
    start: int
    end: int
    heading: str
    text: str
    dataset: str = "stage-1.1-v1"


class Embedder(Protocol):
    identity: str

    def encode(self, texts: list[str]) -> NDArray[np.float32]: ...


class Reranker(Protocol):
    identity: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class VectorStore(Protocol):
    def build(self, collection: str, chunks: list[Chunk], vectors: NDArray) -> None: ...
    def validate(self, collection: str, chunks: list[Chunk]) -> None: ...
    def search(
        self, collection: str, vector: list[float], limit: int
    ) -> list[tuple[str, float]]: ...
