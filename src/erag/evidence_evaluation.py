"""Separate chunk relevance from required-fact evidence coverage.

An alternative is an AND of source spans; alternatives for a fact are ORed. More
equivalent passages never increase the required-fact denominator. This measures
source witnesses, not semantic answer correctness or exhaustive relevance.
"""

import hashlib
from pathlib import Path

from .evaluation import ranking_metrics


def verify_frozen_files(manifest):
    for name, expected in manifest.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen input changed: {name}")


def validate_pool(pool, snapshot):
    if pool["corpus_hash"] != snapshot["corpus_hash"]:
        raise ValueError("evidence pool belongs to a different corpus")
    docs = {d["id"]: d for d in snapshot["documents"]}
    if not pool["cases"]:
        raise ValueError("evidence pool requires cases")
    seen = set()
    for case in pool["cases"]:
        if case["id"] in seen or not case["query"].strip():
            raise ValueError("duplicate or empty evidence case")
        seen.add(case["id"])
        fact_ids = set()
        for fact in case["facts"]:
            if fact["id"] in fact_ids or not fact["alternatives"]:
                raise ValueError("fact requires a unique ID and source alternatives")
            fact_ids.add(fact["id"])
            for alternative in fact["alternatives"]:
                if not alternative["spans"]:
                    raise ValueError("empty source alternative")
                for span in alternative["spans"]:
                    doc = docs.get(span["document_id"])
                    if (
                        doc is None
                        or span["version"] != doc["version"]
                        or not 0 <= span["start"] < span["end"] <= len(doc["text"])
                        or not span["text"].strip()
                        or doc["text"][span["start"] : span["end"]] != span["text"]
                    ):
                        raise ValueError("source witness does not match frozen document")


def chunk_spans(chunks, documents):
    spans = []
    for c in chunks:
        doc = documents.get(c["document_id"])
        if (
            doc is None
            or c["version"] != doc["version"]
            or not 0 <= c["start"] < c["end"] <= len(doc["text"])
            or doc["text"][c["start"] : c["end"]] != c["text"]
        ):
            raise ValueError("context is stale or not an original source slice")
        spans.append((c["document_id"], c["start"], c["end"]))
    return spans


def quote_spans(entries, sources, documents):
    by_id = {s["source_id"]: s["chunk"] for s in sources}
    if len(by_id) != len(sources):
        raise ValueError("duplicate source IDs")
    chunk_spans(list(by_id.values()), documents)
    intervals = []
    for entry in entries:
        chunk = by_id.get(entry["source_id"])
        quote = entry["quote"]
        if chunk is None or not isinstance(quote, str) or not quote or quote not in chunk["text"]:
            raise ValueError("visible fact is outside its supplied source")
        start = chunk["start"] + chunk["text"].index(quote)
        expected = {"document_id": chunk["document_id"], "start": start, "end": start + len(quote)}
        if any(key in entry and entry[key] != value for key, value in expected.items()):
            raise ValueError("citation metadata disagrees with its source quote")
        intervals.append((chunk["document_id"], start, start + len(quote)))
    return intervals


def contains_span(intervals, span):
    cursor = span["start"]
    for doc, a, b in sorted(intervals, key=lambda row: (row[1], row[2])):
        if doc != span["document_id"] or b <= cursor:
            continue
        if a > cursor:
            return False
        cursor = max(cursor, b)
        if cursor >= span["end"]:
            return True
    return False


def fact_coverage(facts, intervals):
    if not facts:
        return {"fact_recall": None, "all_facts_present": None, "facts": []}
    rows = []
    for f in facts:
        covered = any(
            all(contains_span(intervals, s) for s in a["spans"]) for a in f["alternatives"]
        )
        rows.append({"id": f["id"], "covered": covered})
    return {
        "fact_recall": sum(r["covered"] for r in rows) / len(rows),
        "all_facts_present": all(r["covered"] for r in rows),
        "facts": rows,
    }


def analyze_evidence(pool, snapshot, ranking, answers):
    validate_pool(pool, snapshot)
    if (
        ranking["corpus_hash"] != snapshot["corpus_hash"]
        or answers["corpus_hash"] != snapshot["corpus_hash"]
    ):
        raise ValueError("reports and witnesses must use the same corpus")
    if (
        ranking["chunk_hash"] != snapshot["chunk_hash"]
        or answers["chunk_hash"] != snapshot["chunk_hash"]
    ):
        raise ValueError("reports must use the same canonical chunks")
    docs = {d["id"]: d for d in snapshot["documents"]}
    chunks = {c["id"]: c for c in snapshot["chunks"]}
    ranked_rows = [r for r in ranking["queries"] if r["mode"] == "hybrid" and r["reranked"]]
    ranked = {r["case_id"]: r for r in ranked_rows}
    answered = {r["id"]: r for r in answers["cases"]}
    if len(ranked) != len(ranked_rows) or len(answered) != len(answers["cases"]):
        raise ValueError("duplicate report case IDs")
    if set(ranked) != {c["id"] for c in pool["cases"]} or set(answered) != set(ranked):
        raise ValueError("every case requires ranking, witnesses and answer context")
    rows = []
    for case in pool["cases"]:
        r, a = ranked[case["id"]], answered[case["id"]]
        if r["diagnostics"]["query"] != case["query"] or a["query"] != case["query"]:
            raise ValueError("question/context differs between reports")
        ordered = [cid for cid, _ in r["diagnostics"]["final_ranking"]]
        if len(ordered) != len(set(ordered)) or any(cid not in chunks for cid in ordered):
            raise ValueError("ranking contains duplicate or unknown chunks")
        relevance = {}
        for e in case["relevance_evidence"]:
            doc = docs[e["document_id"]]
            if e["text"] not in doc["text"]:
                raise ValueError("reviewed relevance evidence not in source")
            start = doc["text"].index(e["text"])
            for c in chunks.values():
                if (
                    c["document_id"] == doc["id"]
                    and c["start"] <= start
                    and c["end"] >= start + len(e["text"])
                ):
                    relevance[c["id"]] = max(relevance.get(c["id"], 0), e["grade"])
        result = a["result"]
        stored_sources = result.get("sources", [])
        model_source_ids = result.get("model_source_ids")
        if model_source_ids is not None and (
            len(model_source_ids) != len(set(model_source_ids))
            or not set(model_source_ids) <= {s["source_id"] for s in stored_sources}
        ):
            raise ValueError("invalid pre-completion model source identifiers")
        expanded = chunk_spans(
            [
                s["chunk"]
                for s in stored_sources
                if model_source_ids is None or s["source_id"] in model_source_ids
            ],
            docs,
        )
        visible = (
            quote_spans(result["catalog"], result["sources"], docs)
            if "catalog" in result
            else expanded
        )
        selected = quote_spans(
            [e for c in result["claims"] for e in c["citations"]], result.get("sources", []), docs
        )
        stages = {
            f"raw_top{k}": fact_coverage(
                case["facts"], chunk_spans([chunks[c] for c in ordered[:k]], docs)
            )
            for k in (1, 3, 5, 10)
        }
        stages.update(
            expanded_context=fact_coverage(case["facts"], expanded),
            model_visible=fact_coverage(case["facts"], visible),
            selected_quotes=fact_coverage(case["facts"], selected),
        )
        if case["facts"] and "sources" not in result:
            for stage in ("expanded_context", "model_visible"):
                stages[stage] = {
                    "fact_recall": None,
                    "all_facts_present": None,
                    "facts": [{"id": f["id"], "covered": None} for f in case["facts"]],
                    "unavailable_reason": "Answer failure did not preserve the supplied context.",
                }
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "original_raw_chunk_metrics": r["chunk_metrics"],
                "reviewed_pool_raw_chunk_metrics": ranking_metrics(ordered, relevance, [5, 10]),
                "coverage": stages,
            }
        )

    def aggregate(stage):
        required = [r for r, case in zip(rows, pool["cases"], strict=True) if case["facts"]]
        values = [
            r["coverage"][stage] for r in rows if r["coverage"][stage]["fact_recall"] is not None
        ]
        return {
            "questions": len(values),
            "questions_with_requirements": len(required),
            "unmeasured_case_ids": [
                r["id"] for r in required if r["coverage"][stage]["fact_recall"] is None
            ],
            "macro_fact_recall": sum(v["fact_recall"] for v in values) / len(values)
            if values
            else None,
            "all_facts_present_rate": sum(v["all_facts_present"] for v in values) / len(values)
            if values
            else None,
            "conservative_fact_recall_including_unmeasured_as_zero": sum(
                v["fact_recall"] for v in values
            )
            / len(required)
            if required
            else None,
            "missing_case_ids": [
                r["id"] for r in rows if r["coverage"][stage]["all_facts_present"] is False
            ],
        }

    return {
        "pool_version": pool["version"],
        "pool_status": pool["status"],
        "exhaustive": pool["exhaustive"],
        "summary": {stage: aggregate(stage) for stage in rows[0]["coverage"]},
        "cases": rows,
        "limitations": [
            "Source-witness coverage is not semantic answer correctness; unlabelled equivalent evidence may exist.",
            "Missing exact witnesses do not establish an incorrect answer; semantic review is separate.",
            "Reviewed alternatives are a separate, non-exhaustive pool; original labels/targets are unchanged.",
            "Chunk recall counts relevant chunks; fact recall counts requirements, ORing equivalent passages.",
            "Expanded stored sources are not necessarily all shown to the model; model_visible uses the exact saved fact catalog.",
            "Deterministic section completion can add source context after generation; it is excluded from expanded_context/model_visible and included in final selected_quotes.",
            "Missing context on provider failure is unmeasured, not proof of retrieval failure; conservative recall counts it as zero.",
        ],
    }
