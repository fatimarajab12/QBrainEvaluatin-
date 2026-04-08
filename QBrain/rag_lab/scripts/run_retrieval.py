"""
CLI: build FAISS index for one document and run top-k retrieval (same idea as stage 2 + 3 notebooks).

Usage (from rag_lab/):
  python scripts/run_retrieval.py --doc data/srs/sample_srs_a.txt --query "How does password reset work?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from qbrain_rag.application.chunking import chunk_text  # noqa: E402
from qbrain_rag.infrastructure.vector_store import retrieve_top_k  # noqa: E402
from qbrain_rag.infrastructure.document_loaders import load_document  # noqa: E402
from qbrain_rag.infrastructure.vector_store import build_faiss_store  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Index one SRS and print top-k chunks.")
    p.add_argument("--doc", type=Path, required=True, help="Path to .txt or .pdf")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()
    doc = args.doc.resolve()
    if not doc.is_file():
        raise SystemExit(f"File not found: {doc}")
    text = load_document(str(doc))
    chunks = chunk_text(text)
    metadata = [{"source_file": doc.name, "chunk_id": i + 1} for i in range(len(chunks))]
    store = build_faiss_store(chunks, metadata)
    docs = retrieve_top_k(store, args.query, k=args.k)
    for i, d in enumerate(docs, 1):
        print(f"\n--- {i} --- metadata={d.metadata}")
        print(d.page_content[:500])


if __name__ == "__main__":
    main()
