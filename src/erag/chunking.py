"""Paragraph/heading-aware word budgets with exact source character offsets."""

import re
from dataclasses import replace
from itertools import pairwise
from uuid import NAMESPACE_URL, uuid5

from .types import Chunk, Document


def chunk_document(
    doc: Document, max_words: int, overlap_words: int, section_boundaries: bool = False
) -> list[Chunk]:
    if max_words < 1 or not 0 <= overlap_words < max_words:
        raise ValueError("invalid chunk budget")
    if section_boundaries:
        # Keep the title/metadata with the first section, and never overlap into a neighbour.
        headings = list(re.finditer(r"(?m)^#{2,6} .+$", doc.text))
        boundaries = [0] + [h.start() for h in headings[1:]] + [len(doc.text)]
        chunks = []
        all_headings = list(re.finditer(r"(?m)^#{1,6} (.+)$", doc.text))
        for a, b in pairwise(boundaries):
            part = replace(doc, text=doc.text[a:b])
            for c in chunk_document(part, max_words, overlap_words):
                start, end = c.start + a, c.end + a
                heading = next(
                    (h.group(1) for h in reversed(all_headings) if h.start() <= start), ""
                )
                cid = str(uuid5(NAMESPACE_URL, f"{doc.id}:{doc.version}:{start}:{end}"))
                chunks.append(
                    replace(c, id=cid, position=len(chunks), start=start, end=end, heading=heading)
                )
        validate_chunks(doc, chunks, max_words)
        return chunks
    words = list(re.finditer(r"\S+", doc.text))
    # Paragraph boundaries are preferred; headings never get stranded at a chunk end.
    paragraph_ends = {m.start() for m in re.finditer(r"\n\s*\n", doc.text)}
    headings = list(re.finditer(r"(?m)^#{1,6} (.+)$", doc.text))
    chunks, start = [], 0
    while start < len(words):
        hard_end = min(start + max_words, len(words))
        end = hard_end
        if hard_end < len(words):
            choices = [
                i
                for i in range(start + max_words // 2, hard_end + 1)
                if i > start
                and any(words[i - 1].end() <= p < words[i].start() for p in paragraph_ends)
            ]
            if choices:
                end = max(choices)
            # If the boundary strands a heading, move it to the next chunk.
            for h in headings:
                if words[start].start() < h.start() < words[end - 1].end() <= h.end():
                    before = [i for i in range(start + 1, end) if words[i].start() == h.start()]
                    if before:
                        end = before[0]
        a, b = words[start].start(), words[end - 1].end()
        heading = next((h.group(1) for h in reversed(headings) if h.start() <= a), "")
        cid = str(uuid5(NAMESPACE_URL, f"{doc.id}:{doc.version}:{a}:{b}"))
        chunks.append(
            Chunk(
                cid,
                doc.id,
                doc.source,
                doc.version,
                len(chunks),
                a,
                b,
                heading,
                doc.text[a:b],
                doc.dataset,
            )
        )
        if end == len(words):
            break
        start = max(start + 1, end - overlap_words)
    validate_chunks(doc, chunks, max_words)
    return chunks


def validate_chunks(doc: Document, chunks: list[Chunk], max_words: int) -> None:
    covered = bytearray(len(doc.text))
    ids, spans = set(), set()
    for i, c in enumerate(chunks):
        if c.id in ids or (c.start, c.end) in spans:
            raise ValueError("duplicate chunk")
        ids.add(c.id)
        spans.add((c.start, c.end))
        expected_id = str(uuid5(NAMESPACE_URL, f"{doc.id}:{doc.version}:{c.start}:{c.end}"))
        if (
            not 0 <= c.start < c.end <= len(doc.text)
            or c.id != expected_id
            or c.document_id != doc.id
            or c.source != doc.source
            or c.version != doc.version
            or c.dataset != doc.dataset
            or c.position != i
            or c.text != doc.text[c.start : c.end]
            or not c.text.strip()
            or len(c.text.split()) > max_words
        ):
            raise ValueError("chunk metadata/content integrity failure")
        covered[c.start : c.end] = b"\1" * (c.end - c.start)
    if any(not covered[i] for i, ch in enumerate(doc.text) if not ch.isspace()):
        raise ValueError("missing source content")
