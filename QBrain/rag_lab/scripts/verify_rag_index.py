"""
Verify end-to-end: load real SRS → chunk → FAISS → confirm docstore holds every chunk text.

Usage (from rag_lab/):
  python scripts/verify_rag_index.py
  python scripts/verify_rag_index.py --save-cache   # optional disk persistence test

Requires OPENAI_API_KEY for embeddings.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from qbrain_rag.application.chunking import chunk_text  # noqa: E402
from qbrain_rag.config.notebook_defaults import resolve_default_srs_path  # noqa: E402
from qbrain_rag.config.settings import get_settings  # noqa: E402
from qbrain_rag.infrastructure.document_loaders import load_document  # noqa: E402
from qbrain_rag.infrastructure.vector_store import (  # noqa: E402
    chunk_texts_materialized_in_store,
    indexed_document_count,
    load_faiss_store,
    save_faiss_store,
    build_faiss_store,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Verify FAISS indexes all SRS chunks with full text in docstore.")
    p.add_argument("--save-cache", action="store_true", help="Save/load FAISS round-trip under a temp folder")
    args = p.parse_args()

    if not get_settings().openai_api_key:
        raise SystemExit("Set OPENAI_API_KEY in rag_lab/.env")

    srs = resolve_default_srs_path(ROOT)
    print("SRS:", srs)
    text = load_document(str(srs))
    chunks = chunk_text(text)
    meta = [{"source_file": srs.name, "chunk_id": i + 1} for i in range(len(chunks))]
    print(f"Ingestion: {len(text):,} chars -> {len(chunks)} chunks")

    store = build_faiss_store(chunks, meta)
    n = indexed_document_count(store)
    print(f"FAISS vectors + docstore entries: {n}")
    ok = chunk_texts_materialized_in_store(store, chunks)
    print("All chunk texts present in vector store docstore:", "OK" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)

    probe = store.similarity_search("processing icon function", k=1)
    preview = probe[0].page_content[:120].replace("\n", " ")
    preview = preview.encode("ascii", errors="replace").decode("ascii")
    print("Sample retrieval preview:", preview, "...")

    if args.save_cache:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "faiss_test"
            save_faiss_store(store, path)
            loaded = load_faiss_store(path)
            n2 = indexed_document_count(loaded)
            ok2 = chunk_texts_materialized_in_store(loaded, chunks)
            print(f"Round-trip load: entries={n2}, chunks match docstore: {'OK' if ok2 else 'FAIL'}")
            if not ok2 or n2 != n:
                raise SystemExit(1)

    print("verify_rag_index: all checks passed.")


if __name__ == "__main__":
    main()
