"""Bounded second-stage fact search inside the top two retrieved documents.

This supplements answer evidence; it does not rewrite canonical chunk rankings.
Every supplied fact remains an exact source slice with its own provenance.
"""

from uuid import NAMESPACE_URL, uuid5

from .evidence_answer import evidence_catalog, rank_catalog


def search_document_facts(question, retrieval, snapshot, reranker, cfg):
    if reranker is None:
        raise ValueError("document fact search requires a reranker")
    anchors = {}
    for row in retrieval["results"]:
        chunk = row["chunk"]
        anchors.setdefault(chunk["document_id"], chunk)
        if len(anchors) == 2:
            break
    docs = {d["id"]: d for d in snapshot["documents"]}
    sources = []
    for key, anchor in anchors.items():
        doc = docs[key]
        sources.append(
            {
                "source_id": key,
                "chunk": {
                    **anchor,
                    "text": doc["text"],
                    "start": 0,
                    "end": len(doc["text"]),
                    "starts_in_code": False,
                },
            }
        )
    catalog = evidence_catalog(sources)
    selected = rank_catalog(
        question,
        catalog,
        reranker,
        cfg.fact_candidates or cfg.context_chunks,
        cfg.fact_section_coverage,
        cfg.fact_lexical_coverage,
    )
    rows = []
    for rank, fact in enumerate(selected, 1):
        doc = docs[fact["source_id"]]
        start, end = fact["start"], fact["end"]
        chunk = {
            **anchors[doc["id"]],
            "id": str(uuid5(NAMESPACE_URL, f"{doc['id']}:{doc['version']}:{start}:{end}")),
            "start": start,
            "end": end,
            "text": fact["quote"],
            "heading": fact["heading"],
        }
        rows.append({"chunk": chunk, "final_rank": rank})
    return {**retrieval, "results": rows}
