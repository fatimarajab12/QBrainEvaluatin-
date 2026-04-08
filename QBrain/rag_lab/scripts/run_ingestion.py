"""
CLI: load SRS (PDF/TXT), chunk, print stats — mirrors `notebooks/01_ingestion.ipynb`.

Usage (from rag_lab/):
  python scripts/run_ingestion.py data/srs/sample_srs_a.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from qbrain_rag.application.chunking import chunk_text  # noqa: E402
from qbrain_rag.config.settings import get_settings  # noqa: E402
from qbrain_rag.infrastructure.document_loaders import load_document  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Load document, chunk, print statistics.")
    p.add_argument("doc", type=Path, help="Path to .txt or .pdf")
    args = p.parse_args()
    doc = args.doc.resolve()
    if not doc.is_file():
        raise SystemExit(f"File not found: {doc}")
    text = load_document(str(doc))
    chunks = chunk_text(text)
    s = get_settings()
    print(f"File: {doc.name}")
    print(f"Characters: {len(text)}")
    print(f"Chunks: {len(chunks)} (chunk_size={s.chunk_size}, overlap={s.chunk_overlap})")
    print("--- First chunk (preview) ---\n")
    print(chunks[0][:600])


if __name__ == "__main__":
    main()
