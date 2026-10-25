"""Reciprocal rank fusion: only rank positions, never incompatible raw scores."""

import math
from collections import defaultdict


def rrf(
    rankings: list[list[tuple[str, float]]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    if k <= 0:
        raise ValueError("RRF constant must be positive")
    weights = [1.0] * len(rankings) if weights is None else weights
    if len(weights) != len(rankings) or any(
        type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights
    ):
        raise ValueError("RRF weights must match rankings and be finite positive numbers")
    scores = defaultdict(float)
    for ranking, weight in zip(rankings, weights, strict=True):
        seen = set()
        for rank, (cid, _) in enumerate(ranking, 1):
            if cid in seen:
                raise ValueError("duplicate candidate in ranking")
            seen.add(cid)
            scores[cid] += weight / (k + rank)
    return sorted(scores.items(), key=lambda row: (-row[1], row[0]))
