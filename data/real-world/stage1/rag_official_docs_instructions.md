# Official documentation RAG validation set

This is a URL manifest, not downloaded documentation. All URLs point to official vendor domains. Some exact page paths have not been independently verified and must be checked for HTTP 200, correct page content, and redirects before ingestion. Do not silently substitute another page.

## Codex instructions

1. Read `rag_official_docs_30.csv`, fetch each page from the publisher, and verify the final URL and title. Report failures rather than fabricate text.
2. Extract only the main article content into UTF-8 text files. Preserve headings, command blocks, tables where practical, source URL, retrieval date, product version and content hash in per-document metadata. Strip menus, cookie notices and navigation.
3. Do not ingest identical or duplicate content as separate documents. Red Hat docs may include chapter-level overlap; report any duplicates.
4. Preserve exact source copies as permitted by publishers' terms for internal testing; do not redistribute large amounts of vendor content.
5. Freeze the corpus and record hashes. Create human-reviewed, independently authored evaluation questions, with source evidence and unsupported questions. Do not generate both evaluation labels and answers from the RAG's own output.
6. Start with 5 documents. Run ranking and answer evaluation; report results and stop for human approval before expanding to 10, 20, then 30.
7. Compare to the existing synthetic benchmark, without modifying its labels or thresholds. Mark external-corpus metrics separately.
