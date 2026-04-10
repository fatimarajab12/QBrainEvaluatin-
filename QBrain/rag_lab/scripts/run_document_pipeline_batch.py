"""
Run ``run_document_pipeline`` on every ``.pdf`` / ``.txt`` under ``data/srs/`` (or ``--srs-dir``).

By default:
- Deletes legacy folders ``results/paper_output_sample`` and ``results/pipeline_runs``
- Recreates ``results/document_pipeline_runs/`` and writes one JSON per source file

Usage (from ``rag_lab/``):
  python scripts/run_document_pipeline_batch.py
  python scripts/run_document_pipeline_batch.py --no-clean --max-features 3
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from qbrain_rag.application.document_pipeline import run_document_pipeline  # noqa: E402


def safe_stem(filename: str) -> str:
    base = Path(filename).stem
    base = re.sub(r"[^\w\-]+", "_", base, flags=re.UNICODE)
    base = base.strip("_") or "output"
    return base[:120]


def main() -> None:
    p = argparse.ArgumentParser(description="Batch document pipeline for all SRS files.")
    p.add_argument("--srs-dir", type=Path, default=ROOT / "data" / "srs", help="Folder with PDF/TXT sources")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "document_pipeline_runs",
        help="Output directory for JSON artifacts",
    )
    p.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete legacy results or wipe out-dir before run",
    )
    p.add_argument("--max-features", type=int, default=None, metavar="N")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--test-context-k", type=int, default=5)
    args = p.parse_args()

    srs_dir = args.srs_dir.resolve()
    out_dir = args.out_dir.resolve()

    if not srs_dir.is_dir():
        raise SystemExit(f"Not a directory: {srs_dir}")

    legacy_dirs = [
        ROOT / "results" / "paper_output_sample",
        ROOT / "results" / "pipeline_runs",
    ]
    if not args.no_clean:
        for d in legacy_dirs:
            if d.exists():
                print(f"[clean] removing {d}", flush=True)
                # Windows: ignore locked files (e.g. log open in editor)
                shutil.rmtree(d, ignore_errors=True)
        if out_dir.exists():
            print(f"[clean] removing {out_dir}", flush=True)
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(srs_dir.glob("*.pdf")) + sorted(srs_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .pdf or .txt files under {srs_dir}")

    log_lines: list[str] = []
    for path in files:
        print(f"\n[batch] ===== {path.name} =====", flush=True)
        result = run_document_pipeline(
            path,
            n_test_context_chunks=args.test_context_k,
            max_features=args.max_features,
            skip_test_cases=args.skip_tests,
            verbose=True,
        )
        out_name = safe_stem(path.name) + ".json"
        out_path = out_dir / out_name
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        n_feat = len(result.get("features", []))
        n_tc = sum(len(f.get("testCases", [])) for f in result.get("features", []))
        msg = f"OK {path.name} -> {out_name} (features={n_feat}, test_cases={n_tc})"
        print(msg, flush=True)
        log_lines.append(msg)

    log_path = out_dir / "_batch_log.txt"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"\n[batch] log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
