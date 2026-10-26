"""Lossless completion of extractive quotes with bounded Markdown section context.

Only document structure and selected source spans are used; no query, reference
answer or model call is involved. Complete small parent sections keep defaults
and alternatives together. Large sections may use the longest closed subsection
prefix that fits the existing claim budget, never an arbitrary text truncation.
"""

import bisect
import copy
import re
import time
from uuid import NAMESPACE_URL, uuid5

from .answering import validate_answer


def section_ranges(text: str, max_chars: int = 2000) -> list[dict]:
    fences = [m.start() for m in re.finditer(r"(?m)^```", text)]
    nodes, path = [], []
    for match in re.finditer(r"(?m)^(#{1,6}) ([^\n]+)$", text):
        if bisect.bisect_left(fences, match.start()) % 2:
            continue
        level = len(match.group(1))
        path = [(depth, title) for depth, title in path if depth < level]
        path.append((level, match.group(2)))
        nodes.append(
            {"start": match.start(), "level": level, "heading": " > ".join(t for _, t in path)}
        )
    for i, node in enumerate(nodes):
        node["end"] = next(
            (n["start"] for n in nodes[i + 1 :] if n["level"] <= node["level"]), len(text)
        )

    def candidate(node, end, priority):
        end = len(text[:end].rstrip())
        quote = text[node["start"] : end]
        if 20 <= len(quote) <= max_chars and "```" not in quote:
            return {
                "start": node["start"],
                "end": end,
                "heading": node["heading"],
                "priority": priority,
            }
        return None

    result = []
    for i, node in enumerate(nodes):
        children = [n for n in nodes[i + 1 :] if n["start"] < node["end"]]
        whole = candidate(node, node["end"], 0 if children else 2)
        if whole:
            result.append(whole)
            continue
        prefixes = [c for child in children if (c := candidate(node, child["end"], 1))]
        if prefixes:
            result.append(max(prefixes, key=lambda c: c["end"]))
    return result


def complete_sections(result: dict, snapshot: dict, max_chars: int = 2000) -> dict:
    if type(max_chars) is not int or not 20 <= max_chars <= 2000:
        raise ValueError("section completion budget must be from 20 through 2000 characters")
    if result["status"] != "answered" or result.get("section_context"):
        return copy.deepcopy(result)
    started = time.perf_counter()
    out = copy.deepcopy(result)
    documents = {d["id"]: d for d in snapshot["documents"]}
    sources = out["sources"]
    out["model_source_ids"] = [s["source_id"] for s in sources]
    source_by_id = {s["source_id"]: s for s in sources}
    ranges, proposed, expansions = {}, [], []
    for claim in out["claims"]:
        if len(claim["citations"]) != 1 or claim["text"] != claim["citations"][0]["quote"]:
            raise ValueError("section completion requires exact extractive claims")
        cite = claim["citations"][0]
        doc = documents.get(cite["document_id"])
        if (
            doc is None
            or cite["version"] != doc["version"]
            or doc["text"][cite["start"] : cite["end"]] != claim["text"]
        ):
            raise ValueError("section completion source is stale or modified")
        if doc["id"] not in ranges:
            ranges[doc["id"]] = section_ranges(doc["text"], max_chars)
        candidates = [
            r for r in ranges[doc["id"]] if r["start"] <= cite["start"] and r["end"] >= cite["end"]
        ]
        chosen = (
            min(candidates, key=lambda r: (r["priority"], r["end"] - r["start"]))
            if candidates
            else None
        )
        if chosen is None:
            # Repair a clipped paragraph even when the enclosing section is too
            # large. Preserve whole paragraph boundaries and exclude code.
            a = doc["text"].rfind("\n\n", 0, cite["start"]) + 2
            if a == 1:
                a = 0
            b = doc["text"].find("\n\n", cite["end"])
            if b < 0:
                b = len(doc["text"])
            paragraph = doc["text"][a:b]
            inside_code = len(re.findall(r"(?m)^```", doc["text"][:a])) % 2
            if 20 <= len(paragraph) <= max_chars and "```" not in paragraph and not inside_code:
                chosen = {"start": a, "end": b, "heading": "Source paragraph"}
        a, b, sid = cite["start"], cite["end"], cite["source_id"]
        if chosen and (chosen["start"] < a or chosen["end"] > b):
            a, b = chosen["start"], chosen["end"]
            sid = f"SC{len(expansions) + 1}"
            while sid in source_by_id:
                sid += "x"
            base = source_by_id[cite["source_id"]]["chunk"]
            position = next(
                (
                    c["position"]
                    for c in snapshot.get("chunks", [])
                    if c["document_id"] == doc["id"] and c["start"] <= a < c["end"]
                ),
                base.get("position", 0),
            )
            chunk = {
                **base,
                "id": str(uuid5(NAMESPACE_URL, f"{doc['id']}:{doc['version']}:{a}:{b}")),
                "start": a,
                "end": b,
                "text": doc["text"][a:b],
                "heading": chosen["heading"],
                "position": position,
                "starts_in_code": False,
            }
            source = {
                "source_id": sid,
                "chunk": chunk,
                "retrieval_rank": None,
                "origin": "bounded_section_completion",
            }
            sources.append(source)
            source_by_id[sid] = source
            expansions.append(
                {
                    "document_id": doc["id"],
                    "original_start": cite["start"],
                    "original_end": cite["end"],
                    "start": a,
                    "end": b,
                }
            )
        proposed.append(
            {
                "document_id": doc["id"],
                "version": doc["version"],
                "start": a,
                "end": b,
                "text": doc["text"][a:b],
                "source_id": sid,
            }
        )
    kept = []
    for i, row in enumerate(proposed):
        covered = any(
            j != i
            and other["document_id"] == row["document_id"]
            and other["version"] == row["version"]
            and other["start"] <= row["start"]
            and other["end"] >= row["end"]
            and (other["end"] - other["start"] > row["end"] - row["start"] or j < i)
            for j, other in enumerate(proposed)
        )
        if not covered:
            kept.append(
                {
                    "text": row["text"],
                    "citations": [{"source_id": row["source_id"], "quote": row["text"]}],
                }
            )
    status, claims = validate_answer({"status": "answered", "claims": kept}, sources)
    out.update(
        status=status,
        claims=claims,
        model_claims=copy.deepcopy(result["claims"]),
        answer="\n".join(
            c["text"] + " " + " ".join(f"[{e['source_id']}]" for e in c["citations"])
            for c in claims
        ),
        section_context={
            "strategy": "bounded_source_section_supersets",
            "model_called": False,
            "max_claim_chars": max_chars,
            "expansions": expansions,
            "original_claim_count": len(result["claims"]),
            "final_claim_count": len(claims),
        },
    )
    out["section_context_ms"] = (time.perf_counter() - started) * 1000
    out["elapsed_ms"] = result.get("elapsed_ms", 0) + out["section_context_ms"]
    return out
