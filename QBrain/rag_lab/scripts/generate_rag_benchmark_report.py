"""Generate visual/statistical report for rag_benchmark outputs.

Run from rag_lab:
  python scripts/generate_rag_benchmark_report.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    base = Path("data/outputs/evaluation/rag_benchmark")
    out = base / "report"
    out.mkdir(parents=True, exist_ok=True)

    overall = pd.read_csv(base / "overall_summary.csv")
    by_category = pd.read_csv(base / "diagnostics" / "metrics_by_category.csv")
    failure = pd.read_csv(base / "diagnostics" / "failure_breakdown.csv")
    e2e_by_q = pd.read_csv(base / "e2e" / "e2e_by_question.csv")
    gen_by_q = pd.read_csv(base / "generation" / "generation_by_question.csv")
    ret_by_srs = pd.read_csv(base / "retrieval" / "retrieval_summary_by_srs.csv")
    e2e_by_srs = pd.read_csv(base / "e2e" / "e2e_summary_by_srs.csv")

    # 1) Overall metrics bar chart
    overall_map = dict(zip(overall["metric"], overall["value"]))
    keys = [
        "retrieval_hit@5",
        "retrieval_mrr",
        "generation_avg_similarity",
        "e2e_success_rate",
    ]
    labels = ["Hit@5", "MRR", "Gen Similarity", "E2E Success"]
    vals = [float(overall_map.get(k, 0.0)) for k in keys]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, vals)
    plt.ylim(0, 1.0)
    plt.title("Overall RAG Benchmark Metrics")
    plt.ylabel("Score")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    overall_png = out / "overall_metrics.png"
    plt.savefig(overall_png, dpi=160)
    plt.close()

    # 2) Failure breakdown pie
    plt.figure(figsize=(6, 6))
    plt.pie(failure["count"], labels=failure["failure_type"], autopct="%1.1f%%", startangle=90)
    plt.title("Failure Breakdown")
    plt.tight_layout()
    failure_png = out / "failure_breakdown.png"
    plt.savefig(failure_png, dpi=160)
    plt.close()

    # 3) Similarity distribution
    plt.figure(figsize=(8, 4.5))
    plt.hist(gen_by_q["similarity"], bins=15)
    plt.axvline(0.65, linestyle="--")
    plt.title("Generation Similarity Distribution")
    plt.xlabel("Similarity")
    plt.ylabel("Questions")
    plt.tight_layout()
    sim_png = out / "similarity_distribution.png"
    plt.savefig(sim_png, dpi=160)
    plt.close()

    # 4) Per-SRS E2E
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(e2e_by_srs["srs_file"], e2e_by_srs["e2e_success_rate"])
    plt.ylim(0, 1.0)
    plt.xticks(rotation=25, ha="right")
    plt.title("E2E Success by SRS")
    plt.ylabel("Success rate")
    plt.tight_layout()
    srs_png = out / "e2e_by_srs.png"
    plt.savefig(srs_png, dpi=160)
    plt.close()

    # 5) Category radar-like table chart (bar)
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(by_category["category"], by_category["e2e_success_rate"])
    plt.ylim(0, 1.0)
    plt.title("E2E Success by Question Category")
    plt.ylabel("Success rate")
    plt.tight_layout()
    cat_png = out / "e2e_by_category.png"
    plt.savefig(cat_png, dpi=160)
    plt.close()

    # Statistical summary for report
    total_q = int(overall_map.get("questions", len(e2e_by_q)))
    pass_count = int((e2e_by_q["failure_type"] == "pass").sum())
    fail_count = total_q - pass_count
    low_band = int(((gen_by_q["similarity"] >= 0.55) & (gen_by_q["similarity"] < 0.65)).sum())
    high_band = int((gen_by_q["similarity"] >= 0.65).sum())
    very_low = int((gen_by_q["similarity"] < 0.55).sum())

    top5_worst = gen_by_q.sort_values("similarity", ascending=True).head(5)[
        ["question_id", "srs_file", "similarity", "generated_answer"]
    ]

    md = out / "report.md"
    lines: list[str] = []
    lines.append("# RAG Benchmark Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Questions evaluated: **{total_q}**")
    lines.append(f"- Retrieval Hit@5: **{overall_map.get('retrieval_hit@5', 0.0):.3f}**")
    lines.append(f"- Retrieval MRR: **{overall_map.get('retrieval_mrr', 0.0):.3f}**")
    lines.append(f"- Generation accuracy (threshold 0.65): **{_pct(float(overall_map.get('generation_accuracy_sim>=0.65', 0.0)))}**")
    lines.append(f"- Generation average similarity: **{overall_map.get('generation_avg_similarity', 0.0):.3f}**")
    lines.append(f"- End-to-end success: **{_pct(float(overall_map.get('e2e_success_rate', 0.0)))}**")
    lines.append("")
    lines.append("## Key Diagnostics")
    lines.append(f"- Pass count: **{pass_count}**")
    lines.append(f"- Fail count: **{fail_count}**")
    lines.append(f"- Similarity < 0.55: **{very_low}** questions")
    lines.append(f"- Similarity 0.55-0.65: **{low_band}** questions")
    lines.append(f"- Similarity >= 0.65: **{high_band}** questions")
    lines.append("")
    lines.append("## Visuals")
    lines.append(f"- Overall metrics: `{overall_png}`")
    lines.append(f"- Failure breakdown: `{failure_png}`")
    lines.append(f"- Similarity distribution: `{sim_png}`")
    lines.append(f"- E2E by SRS: `{srs_png}`")
    lines.append(f"- E2E by category: `{cat_png}`")
    lines.append("")
    lines.append("## Lowest Similarity Questions (Top 5)")
    lines.append("")
    lines.append("| question_id | srs_file | similarity |")
    lines.append("|---|---|---:|")
    for _, r in top5_worst.iterrows():
        lines.append(f"| {r['question_id']} | {r['srs_file']} | {float(r['similarity']):.3f} |")

    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {md}")
    print(f"Saved visuals in: {out}")
    print(f"Saved files: {', '.join(sorted(p.name for p in out.glob('*')))}")


if __name__ == "__main__":
    main()
