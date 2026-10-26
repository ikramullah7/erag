"""Retrieval-only heading context and exact-span answer context expansion."""

import re
from uuid import NAMESPACE_URL, uuid5


def mark_code_fences(sources, snapshot):
    documents = {d["id"]: d["text"] for d in snapshot["documents"]}
    for source in sources:
        chunk = dict(source["chunk"])
        prefix = documents[chunk["document_id"]][: chunk["start"]]
        chunk["starts_in_code"] = bool(len(re.findall(r"(?m)^```", prefix)) % 2)
        source["chunk"] = chunk
    return sources


def retrieval_text(chunk, document_text):
    headings = []
    for match in re.finditer(r"(?m)^(#{1,6}) (.+)$", document_text[: chunk.end]):
        if match.start() > chunk.start:
            break
        level = len(match.group(1))
        headings = [(depth, title) for depth, title in headings if depth < level]
        headings.append((level, match.group(2)))
    path = " > ".join(title for _, title in headings)
    return f"Document/section: {path}\n\n{chunk.text}"


def expand_context(retrieval, snapshot, neighbors, max_words):
    """Expand ranked anchors without changing indexed chunks or source offsets.

    Preserve anchor priority, merge overlapping windows, and use original document
    slices rather than joining overlapping chunk strings (which duplicates text).
    """
    if not neighbors:
        return retrieval
    docs = {d["id"]: d for d in snapshot["documents"]}
    chunks = {}
    for chunk in snapshot["chunks"]:
        chunks.setdefault(chunk["document_id"], []).append(chunk)
    for group in chunks.values():
        group.sort(key=lambda c: c["position"])
    spans = []
    for row in retrieval["results"]:
        anchor = row["chunk"]
        doc = docs[anchor["document_id"]]
        group = chunks[anchor["document_id"]]
        index = next(i for i, c in enumerate(group) if c["id"] == anchor["id"])
        start, end = anchor["start"], anchor["end"]
        # Prefer the preceding context: a definition/condition often precedes a list item.
        for distance in range(1, neighbors + 1):
            for i in (index - distance, index + distance):
                if 0 <= i < len(group):
                    a, b = min(start, group[i]["start"]), max(end, group[i]["end"])
                    if len(doc["text"][a:b].split()) <= max_words:
                        start, end = a, b
        overlapping = [
            s
            for s in spans
            if s["chunk"]["document_id"] == anchor["document_id"]
            and s["chunk"]["start"] < end
            and start < s["chunk"]["end"]
        ]
        if overlapping:
            a = min(start, *(s["chunk"]["start"] for s in overlapping))
            b = max(end, *(s["chunk"]["end"] for s in overlapping))
            if len(doc["text"][a:b].split()) <= max_words:
                target = overlapping[0]
                start, end = a, b
                for old in overlapping[1:]:
                    spans.remove(old)
            elif any(
                s["chunk"]["start"] <= anchor["start"] and s["chunk"]["end"] >= anchor["end"]
                for s in overlapping
            ):
                continue
            else:
                target = {"chunk": dict(anchor), "final_rank": row["final_rank"]}
                spans.append(target)
        else:
            target = {"chunk": dict(anchor), "final_rank": row["final_rank"]}
            spans.append(target)
        target["chunk"].update(
            id=str(uuid5(NAMESPACE_URL, f"{doc['id']}:{doc['version']}:{start}:{end}")),
            start=start,
            end=end,
            text=doc["text"][start:end],
        )
    return {**retrieval, "results": spans}
