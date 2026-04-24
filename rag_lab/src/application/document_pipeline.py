"""Ingest a file → vector store → LLM features → per-feature test cases (top-k RAG)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from application.chunking import chunk_text
from application.feature_extraction import extract_features_from_indexed_chunks
from application.test_case_generation import generate_test_cases_for_feature
from infrastructure.document_loaders import load_document
from infrastructure.vector_store import build_vector_store


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_document_pipeline(
    path: str | Path,
    *,
    n_test_context_chunks: int = 5,
    max_features: int | None = None,
    skip_test_cases: bool = False,
    verbose: bool = True,
    project_id: str | None = None,
) -> dict[str, Any]:
    """
    ``path``: PDF or text file. Feature step uses **all chunks** in order: one LLM pass if the chunk count
    is small, otherwise **segment extraction** (groups of chunks) plus a **merge/deduplication** pass
    (see ``extract_features_from_indexed_chunks``; ``max_context_chars`` caps each segment). Test step
    uses **similarity_search** per feature (``n_test_context_chunks``).

    ``max_features``: deprecated and ignored. The pipeline returns all extracted features.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(p)

    def log(msg: str) -> None:
        if verbose:
            _progress(msg)

    log(f"[doc] Loading: {p}")
    text = load_document(str(p))
    log(f"[doc] Text length: {len(text):,} chars")
    chunks = chunk_text(text)
    log(f"[doc] Chunks: {len(chunks)} — building vector store...")
    pid = project_id.strip() if project_id else None
    metadata = [
        {"source_file": p.name, "chunk_id": i + 1, **({"project_id": pid} if pid else {})}
        for i in range(len(chunks))
    ]
    store = build_vector_store(chunks, metadata)
    log("[doc] Extracting features (LLM)...")

    feat_bundle = extract_features_from_indexed_chunks(store)
    features: list[dict[str, Any]] = feat_bundle["features"]
    log(f"[doc] Features: {len(feat_bundle['features'])} total, using {len(features)} for tests.")
    if skip_test_cases:
        log("[doc] Skipping test case generation.")

    out_features: list[dict[str, Any]] = []
    for i, f in enumerate(features):
        name = (f.get("name") or "?")[:60]
        if skip_test_cases:
            out_features.append({**f, "testCases": []})
            continue
        log(f"[doc] Tests for feature {i + 1}/{len(features)}: {name!r}...")
        gen = generate_test_cases_for_feature(
            store,
            f,
            n_context_chunks=n_test_context_chunks,
        )
        out_features.append({**f, "evidence": gen["evidence"], "testCases": gen["testCases"]})
        log(f"[doc]   → {len(gen['testCases'])} test case(s)")
    log("[doc] Done.")

    return {
        "source_file": p.name,
        "text_length": len(text),
        "chunk_count": len(chunks),
        "features": out_features,
        "metadata": {
            **feat_bundle["metadata"],
            "feature_count": len(out_features),
            "test_cases_skipped": skip_test_cases,
        },
    }
