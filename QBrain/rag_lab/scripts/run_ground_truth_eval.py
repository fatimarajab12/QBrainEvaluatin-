"""Run RAG evaluation from a ground-truth JSON (same logic as notebooks/06_evaluation_metrics.ipynb)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# rag_lab/src on path
_RAG_LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAG_LAB / "src"))

from qbrain_rag.application.chunking import chunk_text  # noqa: E402
from qbrain_rag.application.evaluation import semantic_similarity  # noqa: E402
from qbrain_rag.infrastructure.document_loaders import load_document  # noqa: E402
from qbrain_rag.infrastructure.llm import answer_with_context  # noqa: E402
from qbrain_rag.infrastructure.vector_store import build_faiss_store, retrieve_top_k  # noqa: E402


def _parse_threshold_sweep(raw: str) -> list[float]:
    if not raw.strip():
        return []
    out: list[float] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        val = float(token)
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"Threshold out of range [0,1]: {val}")
        out.append(val)
    return sorted(set(out))


def build_store_for_file(path: Path):
    text = load_document(str(path))
    chunks = chunk_text(text)
    metas = [{"source_file": path.name, "chunk_id": i + 1} for i in range(len(chunks))]
    return build_faiss_store(chunks, metas)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate RAG against retrieval_ground_truth JSON.")
    p.add_argument(
        "--gt",
        type=Path,
        default=_RAG_LAB / "data" / "ground_truth" / "retrieval" / "retrieval_ground_truth.json",
        help="Path to ground truth JSON",
    )
    p.add_argument("--srs-dir", type=Path, default=_RAG_LAB / "data" / "srs", help="Folder containing SRS PDFs")
    p.add_argument("-k", type=int, default=5, help="Top-k retrieval")
    p.add_argument("--threshold", type=float, default=0.72, help="Similarity threshold for gen_correct")
    p.add_argument(
        "--answer-temperature",
        type=float,
        default=0.1,
        help="Generation temperature used for answer creation during evaluation",
    )
    p.add_argument(
        "--evaluation-mode",
        action="store_true",
        help="Use strict short-answer prompt for evaluation generation.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Output file prefix (e.g. 'gparted_' -> evaluation_results_gparted.csv)",
    )
    p.add_argument(
        "--threshold-sweep",
        type=str,
        default="",
        help="Optional comma-separated thresholds (e.g. 0.65,0.68,0.70,0.72,0.75)",
    )
    args = p.parse_args()

    with open(args.gt, encoding="utf-8-sig") as f:
        ground_truth = json.load(f)

    srs_names = {item["srs_file"] for item in ground_truth}
    stores: dict[str, object] = {}
    for name in sorted(srs_names):
        pdf = args.srs_dir / name
        if not pdf.is_file():
            raise FileNotFoundError(f"SRS not found: {pdf}")
        stores[name] = build_store_for_file(pdf)

    results = []
    for item in sorted(ground_truth, key=lambda x: str(x.get("question_id", ""))):
        fname = item["srs_file"]
        store = stores[fname]
        q = item["question"]
        docs = retrieve_top_k(store, q, k=args.k)
        retrieved_files = sorted({d.metadata.get("source_file") for d in docs if d.metadata.get("source_file")})
        answer = answer_with_context(
            q,
            docs,
            temperature=args.answer_temperature,
            evaluation_mode=args.evaluation_mode,
        )
        sim = semantic_similarity(item["expected_answer"], answer)
        rel = item.get("relevant_files", [fname])
        results.append(
            {
                "question_id": item["question_id"],
                "question": q,
                "srs_file": fname,
                "relevant_files": ";".join(rel),
                "retrieved_files_topk": ";".join(retrieved_files),
                "hit_file": any(f in retrieved_files for f in rel),
                "similarity": sim,
                "gen_correct": sim >= args.threshold,
                "generated_answer": answer,
                "expected_answer": item["expected_answer"],
            }
        )

    hit_at_k = sum(1 for r in results if r["hit_file"]) / len(results)
    answer_accuracy = sum(1 for r in results if r["gen_correct"]) / len(results)
    avg_sim = sum(r["similarity"] for r in results) / len(results)

    summary = pd.DataFrame(
        [
            {"metric": f"Hit@{args.k} (relevant file in top-k)", "value": hit_at_k},
            {"metric": f"Answer accuracy (sim >= {args.threshold})", "value": answer_accuracy},
            {"metric": "Average semantic similarity", "value": avg_sim},
        ]
    )

    out_dir = _RAG_LAB / "data" / "outputs"
    tables_dir = _RAG_LAB / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix
    df = pd.DataFrame(results)
    df.to_csv(out_dir / f"{prefix}evaluation_results.csv", index=False)
    summary.to_csv(out_dir / f"{prefix}summary_metrics.csv", index=False)
    rcols = ["question_id", "srs_file", "hit_file", "relevant_files", "retrieved_files_topk"]
    df[rcols].to_csv(tables_dir / f"{prefix}retrieval_table.csv", index=False)
    df[["question_id", "similarity", "gen_correct", "expected_answer", "generated_answer"]].to_csv(
        tables_dir / f"{prefix}generation_table.csv", index=False
    )
    summary.to_csv(tables_dir / f"{prefix}evaluation_summary.csv", index=False)

    sweep = _parse_threshold_sweep(args.threshold_sweep)
    if sweep:
        sweep_rows: list[dict[str, float]] = []
        eval_df = pd.DataFrame(results)
        for t in sweep:
            gen_correct_t = eval_df["similarity"] >= t
            sweep_rows.append(
                {
                    "threshold": float(t),
                    "generation_accuracy": float(gen_correct_t.mean()),
                    "generation_fail_count": float((~gen_correct_t).sum()),
                }
            )
        pd.DataFrame(sweep_rows).to_csv(out_dir / f"{prefix}threshold_sweep.csv", index=False)
        pd.DataFrame(sweep_rows).to_csv(tables_dir / f"{prefix}threshold_sweep.csv", index=False)

    print(summary.to_string(index=False), flush=True)
    print(f"\nWrote: {out_dir / (prefix + 'evaluation_results.csv')}", flush=True)


if __name__ == "__main__":
    main()
