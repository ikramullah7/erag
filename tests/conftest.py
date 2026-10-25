import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from erag.config import load_config
from erag.pipeline import build_index
from erag.storage import QdrantStore


class FakeEmbedder:
    """Deterministic test double, never used for reported retrieval evaluation."""

    identity = "test-double"

    def encode(self, texts):
        vectors = []
        for t in texts:
            v = np.array(
                [1 + len(t), 1 + t.lower().count("certificate"), 1 + t.lower().count("dns")],
                dtype=np.float32,
            )
            vectors.append(v / np.linalg.norm(v))
        return np.array(vectors)


class FakeReranker:
    identity = "test-double"

    def score(self, query, texts):
        return [float(t.lower().count("certificate")) for t in texts]


@pytest.fixture
def config(tmp_path):
    source = tmp_path / "documents"
    shutil.copytree(Path("data/stage-1.1"), source)
    return replace(
        load_config("config/default.yaml"), dataset=str(source), state=str(tmp_path / "state")
    )


@pytest.fixture
def indexed(config):
    store = QdrantStore()
    embedder = FakeEmbedder()
    snapshot = build_index(config, embedder, store)
    yield config, store, embedder, snapshot
    store.close()


QUALITY_SUMMARY_KEY = pytest.StashKey[dict]()


def pytest_terminal_summary(terminalreporter):
    from erag.evaluation import format_target_table

    summary = terminalreporter.config.stash.get(QUALITY_SUMMARY_KEY, None)
    if summary is not None:
        terminalreporter.write_sep("=", "Stage 1.1 chunk retrieval quality")
        if summary.get("_quality_context"):
            terminalreporter.write_line(summary["_quality_context"])
        terminalreporter.write_line(format_target_table(summary))
