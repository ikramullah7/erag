"""Qdrant local/server adapter with payload-level consistency validation."""

import math
from dataclasses import asdict

import numpy as np
from qdrant_client import QdrantClient, models

from .types import Chunk


class QdrantStore:
    def __init__(self, path: str | None = None, url: str | None = None):
        self.is_local = not bool(url)
        self.client = (
            QdrantClient(url=url, timeout=30)
            if url
            else QdrantClient(path=path)
            if path
            else QdrantClient(":memory:")
        )

    def close(self) -> None:
        self.client.close()

    def build(self, collection: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if (
            vectors.ndim != 2
            or len(vectors) != len(chunks)
            or not len(chunks)
            or not np.isfinite(vectors).all()
            or np.any(np.linalg.norm(vectors, axis=1) == 0)
        ):
            raise ValueError("invalid embedding matrix")
        self.client.create_collection(
            collection,
            vectors_config=models.VectorParams(
                size=vectors.shape[1], distance=models.Distance.COSINE
            ),
        )
        try:
            self.client.upsert(
                collection,
                points=[
                    models.PointStruct(id=c.id, vector=v.tolist(), payload=asdict(c))
                    for c, v in zip(chunks, vectors, strict=True)
                ],
                wait=True,
            )
            self.validate(collection, chunks)
        except Exception:
            self.client.delete_collection(collection)
            raise

    def validate(self, collection: str, chunks: list[Chunk]) -> None:
        actual, offset = {}, None
        while True:
            points, offset = self.client.scroll(
                collection, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            actual.update({str(p.id): p.payload for p in points})
            if offset is None:
                break
        if actual != {c.id: asdict(c) for c in chunks}:
            raise ValueError("BM25/Qdrant chunk collection mismatch")

    def search(self, collection: str, vector: list[float], limit: int) -> list[tuple[str, float]]:
        if not vector or not all(math.isfinite(x) for x in vector):
            raise ValueError("invalid query vector")
        points = self.client.query_points(
            collection,
            query=vector,
            limit=limit,
            search_params=None if self.is_local else models.SearchParams(exact=True),
        ).points
        return sorted(
            [(str(p.id), float(p.score)) for p in points], key=lambda row: (-row[1], row[0])
        )
