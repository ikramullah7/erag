"""Replay cached retrieval ablations; optionally run bounded local-14B diagnostic probes.

Observed questions are development diagnostics now, never a new held-out test.
No index, baseline, labels, model downloads or default configurations are changed.
"""

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from erag.answer_config import load_answer_config
from erag.answering import answer_question
from erag.evaluation import ranking_metrics
from erag.fusion import rrf
from erag.ollama import OllamaGenerator
from erag.pipeline import atomic_json


def candidate_ids(lexical, vector, policy):
    if policy == "lexical":
        return [cid for cid, _ in lexical]
    combined = rrf([lexical, vector], 60, [1, 1])
    if policy == "balanced20":
        combined = combined[:20]
    elif policy != "union":
        raise ValueError("unknown candidate policy")
    return [cid for cid, _ in combined]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise ValueError("preserve previous diagnostics; choose a new output")
    root.mkdir(parents=True)
    run = Path("artifacts/real-world/stage1/run-001")
    ranking = json.loads((run / "heldout.json").read_text())
    answers = json.loads((run / "answers.json").read_text())
    snapshot = json.loads(Path("artifacts/real-world/stage1/index/manifest.json").read_text())
    chunks = {c["id"]: c for c in snapshot["chunks"]}
    lookup = {(r["case_id"], r["mode"], r["reranked"]): r for r in ranking["queries"]}
    original = {r["id"]: r for r in answers["cases"]}
    variants = {key: [] for key in ("lexical", "balanced20", "union")}
    details, rerankings = [], {}
    max_score_delta = 0.0
    for qid, case in original.items():
        h = lookup[qid, "hybrid", True]
        b = lookup[qid, "bm25", True]["diagnostics"]
        v = lookup[qid, "vector", True]["diagnostics"]
        bs, vs = dict(b["final_ranking"]), dict(v["final_ranking"])
        max_score_delta = max(
            max_score_delta, max((abs(bs[c] - vs[c]) for c in bs.keys() & vs.keys()), default=0)
        )
        scores = {**bs, **vs}
        gold = h["relevant_chunks"]
        pools = {}
        for policy, measurements in variants.items():
            ids = candidate_ids(b["components"]["bm25"], v["components"]["cosine"], policy)
            ids = sorted(ids, key=lambda cid: (-scores[cid], cid))
            rerankings[qid, policy] = ids
            pools[policy] = {
                "count": len(ids),
                "label_ranks": [i for i, c in enumerate(ids, 1) if c in gold],
            }
            if gold:
                measurements.append(
                    {
                        **ranking_metrics(ids, gold, [5, 10]),
                        "pool_any": float(bool(set(ids) & gold.keys())),
                    }
                )
        result = case["result"]
        actual = {s["chunk"]["id"] for s in result["sources"]}
        details.append(
            {
                "id": qid,
                "query": case["query"],
                "status": result["status"],
                "hybrid_pool_equals_bm25": {c for c, _ in h["diagnostics"]["initial_ranking"]}
                == set(bs),
                "labeled_chunks": len(gold),
                "labeled_chunks_in_actual_context": len(actual & gold.keys()),
                "actual_context_ranks": [s["retrieval_rank"] for s in result["sources"]],
                "context_was_trimmed": len(result["sources"]) < 6,
                "policies": pools,
                "validation_error": result.get("validation_error"),
            }
        )
    if max_score_delta > 0.001:
        raise ValueError("cached per-pair scores differ; require a fresh controlled reranker run")
    summaries = {
        policy: {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
        for policy, rows in variants.items()
    }
    report = {
        "kind": "observed-set diagnostic; not held-out improvement evidence",
        "input_hashes": {
            str(run / n): hashlib.sha256((run / n).read_bytes()).hexdigest()
            for n in ("heldout.json", "answers.json")
        },
        "method": "Reuse independently measured BGE pair scores from BM25/vector reranked runs; change candidate selection only, not scores or labels.",
        "max_shared_pair_score_delta": max_score_delta,
        "summaries": summaries,
        "cases": details,
        "limitations": [
            "Selected evidence labels are not exhaustive relevance judgments.",
            "Union uses up to forty candidates; balanced20 keeps the twenty-candidate budget.",
            "No fresh retrieval latency or full answer aggregate claimed.",
            "Frozen review rubric has stale inherited prose (15/20 cases and certificates); scoring uses actual 20 answerable/22 total cases, not this prose. Original annotation artifact preserved.",
        ],
    }
    atomic_json(root / "diagnosis.json", report)
    print(
        json.dumps({"candidate_ablations": summaries, "pair_score_delta": max_score_delta}),
        flush=True,
    )
    if not args.generate:
        return

    cfg = load_answer_config("config/answer-real-stage1.yaml")
    generator = OllamaGenerator(cfg)

    class Evidence:
        def __init__(self, retrieval, engine_cfg):
            self.retrieval, self.cfg = retrieval, engine_cfg

        def search(self, *args, **kwargs):
            return deepcopy(self.retrieval)

    from types import SimpleNamespace

    from erag import answering

    original_prompt = answering.SYSTEM_PROMPT
    atomic_addition = """
Additional output discipline: give the direct answer first, then only qualifications needed
for this question. Use one atomic factual assertion per claim; separate different entities
and their definitions into separate claims. Check whether a retrieved exception changes
your answer before asserting a general rule. Do not fill the six-claim limit with background.
Copy each quote as ONE contiguous substring of ONE source, including its exact whitespace;
do not concatenate bullets or sentences separated by omitted text into a single quote.
These instructions do not relax the original evidence, completeness or abstention policy.
"""
    # Mechanism-selected probes, chosen after inspecting failures. Not a validation sample.
    for qid, arm in [
        ("q13", "balanced20"),
        ("q14", "neighbor"),
        ("q05", "atomic"),
        ("q08", "atomic"),
    ]:
        baseline = lookup[qid, "hybrid", True]["diagnostics"]
        retrieval = deepcopy(baseline)
        ids = [cid for cid, _ in baseline["final_ranking"]]
        if arm == "balanced20":
            ids = rerankings[qid, "balanced20"]
        elif arm == "neighbor":
            first = chunks[ids[0]]
            adjacent = sorted(
                (
                    c
                    for c in chunks.values()
                    if c["document_id"] == first["document_id"]
                    and abs(c["position"] - first["position"]) == 1
                ),
                key=lambda c: c["position"],
            )
            ids = list(dict.fromkeys([ids[0], *[c["id"] for c in adjacent], *ids]))
        retrieval["results"] = [
            {"chunk": chunks[cid], "final_rank": i} for i, cid in enumerate(ids, 1)
        ]
        answering.SYSTEM_PROMPT = original_prompt + (atomic_addition if arm == "atomic" else "")
        try:
            result = answer_question(
                Evidence(retrieval, SimpleNamespace(rerank=True, candidates=20)),
                generator,
                cfg,
                original[qid]["query"],
            )
        finally:
            answering.SYSTEM_PROMPT = original_prompt
        output = {
            "id": qid,
            "arm": arm,
            "diagnostic_only": True,
            "baseline_status": original[qid]["result"]["status"],
            "baseline_answer": original[qid]["result"]["answer"],
            "prompt_addition": atomic_addition if arm == "atomic" else None,
            "system_prompt_sha256": hashlib.sha256(
                (original_prompt + (atomic_addition if arm == "atomic" else "")).encode()
            ).hexdigest(),
            "answer_config": asdict(cfg),
            "result": result,
        }
        atomic_json(root / f"probe-{qid}-{arm}.json", output)
        print(
            json.dumps(
                {"id": qid, "arm": arm, "status": result["status"], "answer": result["answer"]}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
