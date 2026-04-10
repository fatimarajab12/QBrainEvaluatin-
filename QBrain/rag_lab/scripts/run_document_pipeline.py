"""
CLI: document → features (single pass or segment+merge) → test cases per feature (multi-query RAG + evidence).

Usage (from rag_lab/, with OPENAI_API_KEY in .env):
  python scripts/run_document_pipeline.py path/to/file.pdf
  python scripts/run_document_pipeline.py path/to/file.pdf --max-features 5 --skip-tests

Outputs JSON to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from qbrain_rag.application.document_pipeline import run_document_pipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Document pipeline: features + test cases per feature.")
    p.add_argument("doc", type=Path, help="Path to .txt or .pdf (or other supported format)")
    p.add_argument(
        "--max-features",
        type=int,
        default=None,
        metavar="N",
        help="Optional: only generate test cases for the first N features (model order). "
        "Default: no limit.",
    )
    p.add_argument("--skip-tests", action="store_true", help="Only extract features")
    p.add_argument("--test-context-k", type=int, default=5, help="Top-k unique chunks per feature for tests")
    p.add_argument("--quiet", action="store_true", help="No progress messages (stderr)")
    args = p.parse_args()
    doc = args.doc.resolve()
    if not doc.is_file():
        raise SystemExit(f"File not found: {doc}")

    result = run_document_pipeline(
        doc,
        n_test_context_chunks=args.test_context_k,
        max_features=args.max_features,
        skip_test_cases=args.skip_tests,
        verbose=not args.quiet,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
