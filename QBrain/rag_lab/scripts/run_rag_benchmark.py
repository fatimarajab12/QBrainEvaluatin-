"""Run a structured RAG benchmark on 3 levels: retrieval, generation, end-to-end.

Folder convention:
  data/benchmarks/rag_eval/questions/retrieval_ground_truth*.json
  data/outputs/evaluation/rag_benchmark/{retrieval,generation,e2e,diagnostics}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

_RAG_LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAG_LAB / "src"))

from qbrain_rag.application.chunking import chunk_text  # noqa: E402
from qbrain_rag.application.evaluation import semantic_similarity  # noqa: E402
from qbrain_rag.infrastructure.document_loaders import load_document  # noqa: E402
from qbrain_rag.infrastructure.llm import answer_with_context  # noqa: E402
from qbrain_rag.infrastructure.vector_store import build_faiss_store, retrieve_top_k  # noqa: E402


def _ensure_dirs(base: Path) -> dict[str, Path]:
    out = {
        "base": base,
        "retrieval": base / "retrieval",
        "generation": base / "generation",
        "e2e": base / "e2e",
        "diagnostics": base / "diagnostics",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def _load_questions(questions_dir: Path) -> list[dict[str, Any]]:
    files = sorted(questions_dir.glob("retrieval_ground_truth*.json"))
    if not files:
        raise FileNotFoundError(f"No retrieval_ground_truth*.json found in {questions_dir}")
    items: list[dict[str, Any]] = []
    for path in files:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            if not all(k in item for k in ("question_id", "question", "srs_file", "expected_answer")):
                continue
            item = dict(item)
            item["category"] = str(item.get("category", "direct"))
            items.append(item)
    return items


def _cap_per_srs(items: list[dict[str, Any]], max_questions: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        grouped[str(it["srs_file"])].append(it)
    out: list[dict[str, Any]] = []
    for srs_name in sorted(grouped):
        ordered = sorted(grouped[srs_name], key=lambda x: str(x.get("question_id", "")))
        out.extend(ordered[:max_questions])
    return out


def _build_store(path: Path):
    text = load_document(str(path))
    chunks = chunk_text(text)
    metas = [{"source_file": path.name, "chunk_id": i + 1} for i in range(len(chunks))]
    return build_faiss_store(chunks, metas)


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _first_relevant_rank(retrieved: list[str], relevant: set[str]) -> int | None:
    for i, name in enumerate(retrieved, start=1):
        if name in relevant:
            return i
    return None


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


def run_benchmark(
    questions: list[dict[str, Any]],
    *,
    srs_dir: Path,
    k: int,
    threshold: float,
    answer_temperature: float,
    evaluation_mode: bool,
) -> pd.DataFrame:
    srs_names = sorted({str(q["srs_file"]) for q in questions})
    stores: dict[str, Any] = {}
    for name in srs_names:
        path = srs_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing SRS file: {path}")
        stores[name] = _build_store(path)

    rows: list[dict[str, Any]] = []
    ordered = sorted(questions, key=lambda x: (str(x["srs_file"]), str(x.get("question_id", ""))))
    for item in ordered:
        qid = str(item["question_id"])
        qtxt = str(item["question"])
        srs_file = str(item["srs_file"])
        category = str(item.get("category", "direct"))
        relevant_list = item.get("relevant_files", [srs_file])
        if not isinstance(relevant_list, list):
            relevant_list = [srs_file]
        relevant = {str(x) for x in relevant_list}
        docs = retrieve_top_k(stores[srs_file], qtxt, k=k)
        retrieved_sources = _unique_in_order([str(d.metadata.get("source_file", "")) for d in docs])
        inter = relevant.intersection(retrieved_sources)
        rank = _first_relevant_rank(retrieved_sources, relevant)

        precision_at_k = len(inter) / max(1, k)
        recall_at_k = len(inter) / max(1, len(relevant))
        hit_at_k = 1.0 if len(inter) > 0 else 0.0
        mrr = 1.0 / rank if rank else 0.0

        answer = answer_with_context(
            qtxt,
            docs,
            temperature=answer_temperature,
            evaluation_mode=evaluation_mode,
        )
        expected = str(item["expected_answer"])
        similarity = semantic_similarity(expected, answer)
        gen_correct = similarity >= threshold

        if hit_at_k == 0:
            failure_type = "retrieval_fail"
        elif not gen_correct:
            failure_type = "generation_fail"
        else:
            failure_type = "pass"

        rows.append(
            {
                "question_id": qid,
                "category": category,
                "srs_file": srs_file,
                "question": qtxt,
                "relevant_files": ";".join(sorted(relevant)),
                "retrieved_files_topk": ";".join(retrieved_sources),
                "retrieved_contexts_json": json.dumps([str(d.page_content) for d in docs], ensure_ascii=False),
                "hit_at_k": hit_at_k,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "mrr": mrr,
                "similarity": similarity,
                "gen_correct": bool(gen_correct),
                "failure_type": failure_type,
                "expected_answer": expected,
                "generated_answer": answer,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structured RAG benchmark (retrieval + generation + e2e).")
    parser.add_argument(
        "--questions-dir",
        type=Path,
        default=_RAG_LAB / "data" / "benchmarks" / "rag_eval" / "questions",
        help="Folder with retrieval_ground_truth*.json files",
    )
    parser.add_argument("--srs-dir", type=Path, default=_RAG_LAB / "data" / "srs")
    parser.add_argument("--max-questions-per-srs", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument(
        "--threshold-sweep",
        type=str,
        default="",
        help="Optional comma-separated thresholds (e.g. 0.65,0.68,0.70,0.72,0.75)",
    )
    parser.add_argument("--answer-temperature", type=float, default=0.1)
    parser.add_argument(
        "--evaluation-mode",
        action="store_true",
        help="Use strict short-answer prompt for evaluation generation.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_RAG_LAB / "data" / "outputs" / "evaluation" / "rag_benchmark",
    )
    args = parser.parse_args()

    dirs = _ensure_dirs(args.out_dir)
    questions = _cap_per_srs(_load_questions(args.questions_dir), max(1, args.max_questions_per_srs))
    df = run_benchmark(
        questions,
        srs_dir=args.srs_dir,
        k=args.k,
        threshold=args.threshold,
        answer_temperature=args.answer_temperature,
        evaluation_mode=args.evaluation_mode,
    )

    # level 1: retrieval
    retrieval_q = df[
        ["question_id", "category", "srs_file", "hit_at_k", "precision_at_k", "recall_at_k", "mrr", "relevant_files", "retrieved_files_topk"]
    ].copy()
    retrieval_q.to_csv(dirs["retrieval"] / "retrieval_by_question.csv", index=False)
    retrieval_srs = (
        retrieval_q.groupby("srs_file", as_index=False)
        .agg(
            questions=("question_id", "count"),
            hit_rate=("hit_at_k", "mean"),
            precision_at_k=("precision_at_k", "mean"),
            recall_at_k=("recall_at_k", "mean"),
            mrr=("mrr", "mean"),
        )
        .sort_values("srs_file")
    )
    retrieval_srs.to_csv(dirs["retrieval"] / "retrieval_summary_by_srs.csv", index=False)

    # level 2: generation
    generation_q = df[
        ["question_id", "category", "srs_file", "similarity", "gen_correct", "expected_answer", "generated_answer"]
    ].copy()
    generation_q.to_csv(dirs["generation"] / "generation_by_question.csv", index=False)
    generation_srs = (
        generation_q.groupby("srs_file", as_index=False)
        .agg(
            questions=("question_id", "count"),
            answer_accuracy=("gen_correct", "mean"),
            avg_similarity=("similarity", "mean"),
            min_similarity=("similarity", "min"),
            max_similarity=("similarity", "max"),
        )
        .sort_values("srs_file")
    )
    generation_srs.to_csv(dirs["generation"] / "generation_summary_by_srs.csv", index=False)

    # level 3: end-to-end
    e2e_q = df[
        ["question_id", "category", "srs_file", "hit_at_k", "gen_correct", "failure_type", "similarity"]
    ].copy()
    e2e_q["e2e_success"] = (e2e_q["hit_at_k"] == 1.0) & (e2e_q["gen_correct"] == True)  # noqa: E712
    e2e_q.to_csv(dirs["e2e"] / "e2e_by_question.csv", index=False)

    e2e_srs = (
        e2e_q.groupby("srs_file", as_index=False)
        .agg(
            questions=("question_id", "count"),
            e2e_success_rate=("e2e_success", "mean"),
            retrieval_hit_rate=("hit_at_k", "mean"),
            generation_accuracy=("gen_correct", "mean"),
        )
        .sort_values("srs_file")
    )
    e2e_srs.to_csv(dirs["e2e"] / "e2e_summary_by_srs.csv", index=False)

    failures = (
        e2e_q.groupby(["failure_type"], as_index=False)
        .agg(count=("question_id", "count"))
        .sort_values("count", ascending=False)
    )
    failures.to_csv(dirs["diagnostics"] / "failure_breakdown.csv", index=False)

    by_category = (
        e2e_q.groupby(["category"], as_index=False)
        .agg(
            questions=("question_id", "count"),
            e2e_success_rate=("e2e_success", "mean"),
            retrieval_hit_rate=("hit_at_k", "mean"),
            generation_accuracy=("gen_correct", "mean"),
            avg_similarity=("similarity", "mean"),
        )
        .sort_values("category")
    )
    by_category.to_csv(dirs["diagnostics"] / "metrics_by_category.csv", index=False)

    overall = pd.DataFrame(
        [
            {"metric": "srs_files", "value": float(df["srs_file"].nunique())},
            {"metric": "questions", "value": float(len(df))},
            {"metric": f"retrieval_hit@{args.k}", "value": float(df["hit_at_k"].mean())},
            {"metric": f"retrieval_precision@{args.k}", "value": float(df["precision_at_k"].mean())},
            {"metric": f"retrieval_recall@{args.k}", "value": float(df["recall_at_k"].mean())},
            {"metric": "retrieval_mrr", "value": float(df["mrr"].mean())},
            {"metric": f"generation_accuracy_sim>={args.threshold}", "value": float(df["gen_correct"].mean())},
            {"metric": "generation_avg_similarity", "value": float(df["similarity"].mean())},
            {
                "metric": "e2e_success_rate",
                "value": float(((df["hit_at_k"] == 1.0) & (df["gen_correct"] == True)).mean()),  # noqa: E712
            },
        ]
    )
    overall.to_csv(dirs["base"] / "overall_summary.csv", index=False)
    df.to_csv(dirs["base"] / "benchmark_full_results.csv", index=False)

    sweep = _parse_threshold_sweep(args.threshold_sweep)
    if sweep:
        rows: list[dict[str, float]] = []
        for t in sweep:
            gen_correct_t = df["similarity"] >= t
            e2e_success_t = (df["hit_at_k"] == 1.0) & gen_correct_t
            rows.append(
                {
                    "threshold": float(t),
                    "generation_accuracy": float(gen_correct_t.mean()),
                    "e2e_success_rate": float(e2e_success_t.mean()),
                    "generation_fail_count": float((~gen_correct_t).sum()),
                    "pass_count": float(e2e_success_t.sum()),
                }
            )
        pd.DataFrame(rows).to_csv(dirs["base"] / "threshold_sweep.csv", index=False)

    print("Saved benchmark outputs to:", dirs["base"])
    print("\nOverall summary:\n", overall.to_string(index=False))


if __name__ == "__main__":
    main()
