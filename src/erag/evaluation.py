"""Evidence-validated ranking metrics and reproducible experiment records."""

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from .config import digest
from .pipeline import Engine, atomic_json


def ranking_metrics(ranked: list[str], grades: dict[str, int], ks: list[int]) -> dict:
    if len(ranked) != len(set(ranked)):
        raise ValueError("ranking contains duplicate IDs")
    positive = {k: v for k, v in grades.items() if v > 0}
    if not positive:
        return {}
    if any(type(g) is not int or g not in {1, 2, 3} for g in positive.values()):
        raise ValueError("relevance grades must be integers from 1 to 3")
    out = {"mrr": next((1 / i for i, key in enumerate(ranked, 1) if key in positive), 0.0)}
    out["hit@1"] = float(any(key in positive for key in ranked[:1]))
    out["hit@3"] = float(any(key in positive for key in ranked[:3]))
    ideal = sorted(positive.values(), reverse=True)
    for k in ks:
        out[f"recall@{k}"] = len(set(ranked[:k]) & positive.keys()) / len(positive)
        dcg = sum(
            (2 ** positive.get(key, 0) - 1) / math.log2(i + 2) for i, key in enumerate(ranked[:k])
        )
        idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
        out[f"ndcg@{k}"] = dcg / idcg
    return out


def load_cases(path: Path, engine: Engine) -> list[dict]:
    cases = json.loads(path.read_text())
    docs = {d["id"]: d["text"] for d in engine.snapshot["documents"]}
    seen = set()
    for case in cases:
        if case["id"] in seen or not case["query"].strip():
            raise ValueError("duplicate case ID or empty query")
        seen.add(case["id"])
        ev_docs, chunk_grades = {}, {}
        for evidence in case["evidence"]:
            doc, text, grade = evidence["document_id"], evidence["text"], evidence["grade"]
            if type(grade) is not int or grade not in {1, 2, 3} or not text.strip():
                raise ValueError("invalid evidence grade or text")
            if doc not in docs or text not in docs[doc]:
                raise ValueError(f"unsupported evidence: {case['id']}")
            ev_docs[doc] = max(ev_docs.get(doc, 0), grade)
            # Use source spans rather than lexical matches in unrelated chunks.
            start = docs[doc].index(text)
            matches = [
                c.id
                for c in engine.chunks
                if c.document_id == doc and c.start <= start and c.end >= start + len(text)
            ]
            if not matches:
                raise ValueError(f"evidence crosses chunk boundary: {case['id']}")
            for cid in matches:
                chunk_grades[cid] = max(chunk_grades.get(cid, 0), grade)
        if ev_docs != case["relevant_documents"]:
            raise ValueError("document grades disagree with evidence")
        if (case["category"] == "no-answer") != (not ev_docs):
            raise ValueError("no-answer label mismatch")
        case["relevant_chunks"] = chunk_grades
    return cases


def environment() -> dict:
    packages = (
        "qdrant-client",
        "rank-bm25",
        "sentence-transformers",
        "torch",
        "transformers",
        "numpy",
        "PyYAML",
        "pytest",
    )
    lock = Path("uv.lock")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    # Include working tree source hash even when HEAD is absent or dirty.
    source = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path("src").rglob("*.py"))
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "lock_hash": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_hash": digest(source),
    }


def evaluate(
    engine: Engine,
    cases_path: Path,
    report_path: Path,
    experiment_id: str,
    split: str,
    strategies: tuple[tuple[str, bool], ...] | None = None,
) -> dict:
    if (
        split
        not in {
            "tuning",
            "heldout",
            "tuning-v2",
            "heldout-v2",
            "tuning-v3",
            "heldout-v3",
            "tuning-v4",
            "heldout-v4",
        }
        or cases_path.name != f"{split}.json"
    ):
        raise ValueError("explicit evaluation split must match the dataset filename")
    strategies = strategies or tuple(
        (m, r) for m in ("bm25", "vector", "hybrid") for r in (False, True)
    )
    allowed = {(m, r) for m in ("bm25", "vector", "hybrid") for r in (False, True)}
    if len(set(strategies)) != len(strategies) or any(s not in allowed for s in strategies):
        raise ValueError("invalid or duplicate evaluation strategies")
    cases = load_cases(cases_path, engine)
    # Warm every code path; load/index times are reported separately.
    for mode, rerank in strategies:
        engine.search("warm up retrieval", mode, rerank)
    rows = []
    for case in cases:
        for mode, rerank in strategies:
            result = engine.search(case["query"], mode, rerank, engine.cfg.candidates)
            ranked_chunks = [cid for cid, _ in result["final_ranking"]]
            ranked_docs = list(
                dict.fromkeys(engine.by_id[cid].document_id for cid in ranked_chunks)
            )
            dm = ranking_metrics(ranked_docs, case["relevant_documents"], [3, 5])
            dm.pop("ndcg@3", None)
            cm = ranking_metrics(ranked_chunks, case["relevant_chunks"], [3, 5, 10])
            cm.pop("ndcg@3", None)
            failed = bool(dm and (dm["recall@3"] < 1 or cm["recall@5"] < 1))
            rows.append(
                {
                    "case_id": case["id"],
                    "category": case["category"],
                    "answerable": bool(dm),
                    "mode": mode,
                    "reranked": rerank,
                    "document_metrics": dm,
                    "chunk_metrics": cm,
                    "relevant_documents": case["relevant_documents"],
                    "relevant_chunks": case["relevant_chunks"],
                    "failed": failed,
                    "diagnostics": result,
                }
            )
    summaries = {}
    for mode in ("bm25", "vector", "hybrid"):
        for rerank in (False, True):
            group = [r for r in rows if r["mode"] == mode and r["reranked"] == rerank]
            if not group:
                continue
            answerable = [r for r in group if r["answerable"]]
            if not answerable:
                raise ValueError("evaluation requires answerable queries")
            means = lambda subset, kind: {
                k: float(np.mean([r[kind][k] for r in subset])) for k in subset[0][kind]
            }
            categories = {}
            for category in sorted({r["category"] for r in answerable}):
                subset = [r for r in answerable if r["category"] == category]
                categories[category] = {
                    "document": means(subset, "document_metrics"),
                    "chunk": means(subset, "chunk_metrics"),
                }
            summaries[f"{mode}{'+rerank' if rerank else ''}"] = {
                "document": means(answerable, "document_metrics"),
                "chunk": means(answerable, "chunk_metrics"),
                "latency_ms_p50": float(
                    np.percentile([r["diagnostics"]["timings"]["total_ms"] for r in group], 50)
                ),
                "latency_ms_p95": float(
                    np.percentile([r["diagnostics"]["timings"]["total_ms"] for r in group], 95)
                ),
                "failed_queries": [r["case_id"] for r in answerable if r["failed"]],
                "no_answer_queries": [r["case_id"] for r in group if not r["answerable"]],
                "answerable_count": len(answerable),
                "by_category": categories,
            }
    report = {
        "experiment_id": experiment_id,
        "split": split,
        "dataset_version": f"stage-{engine.cfg.stage}-v1",
        "created_unix": time.time(),
        "config": asdict(engine.cfg),
        "corpus_hash": engine.snapshot["corpus_hash"],
        "chunk_hash": engine.snapshot["chunk_hash"],
        "labels_hash": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "index_ms": engine.snapshot["index_ms"],
        "chunk_count": len(engine.chunks),
        "environment": environment(),
        "strategies_evaluated": strategies,
        "summary": summaries,
        "queries": rows,
    }
    atomic_json(report_path, report)
    return report


def load_quality_targets(path: str | Path = "config/quality.yaml") -> dict[str, float]:
    targets = yaml.safe_load(Path(path).read_text())
    expected = {"hit@1", "hit@3", "recall@5", "mrr", "ndcg@10"}
    if not isinstance(targets, dict) or set(targets) != expected:
        raise ValueError("quality targets must define exactly the five approved chunk metrics")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
        for v in targets.values()
    ):
        raise ValueError("quality targets must be finite numbers between zero and one")
    return targets


def target_results(summary: dict, targets: dict[str, float] | None = None) -> list[dict]:
    targets = load_quality_targets() if targets is None else targets
    rows = []
    for metric, threshold in targets.items():
        actual = summary["chunk"].get(metric)
        valid = type(actual) in (int, float) and math.isfinite(actual) and 0 <= actual <= 1
        rows.append(
            {
                "metric": metric,
                "target": threshold,
                "result": actual,
                "passed": valid and actual >= threshold,
            }
        )
    return rows


def format_target_table(summary: dict) -> str:
    labels = {
        "hit@1": "Hit@1",
        "hit@3": "Hit@3",
        "recall@5": "Recall@5",
        "mrr": "MRR",
        "ndcg@10": "nDCG@10",
    }
    lines = ["Metric       Target       Our result   Meets target", "-" * 52]
    for row in target_results(summary):
        percent = row["metric"] in {"hit@1", "hit@3", "recall@5"}

        def display(value, percent=percent):
            if value is None:
                return "missing"
            return f"{value * 100:.2f}%" if percent else f"{value:.4f}"

        lines.append(
            f"{labels[row['metric']]:<12} >= {display(row['target']):<9} "
            f"{display(row['result']):<12} {'PASS' if row['passed'] else 'FAIL'}"
        )
    return "\n".join(lines)


def quality_gate(summary: dict, reranked: bool) -> dict:
    checks = {
        "document_recall@3": summary["document"]["recall@3"] >= 0.90,
        "document_mrr": summary["document"]["mrr"] >= 0.85,
        "document_ndcg@5": summary["document"]["ndcg@5"] >= 0.85,
        **{f"chunk_{row['metric']}": row["passed"] for row in target_results(summary)},
        "latency": summary["latency_ms_p95"] <= (5000 if reranked else 2000),
    }
    return {"passed": all(checks.values()), "checks": checks}
