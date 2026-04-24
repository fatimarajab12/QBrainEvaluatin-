from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class QueryResult:
    question_id: str
    question: str
    relevant_file: str
    ranked_files: list[str]
    hit_at_k: float
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate retrieval quality via API endpoints (project upload + project retrieval)."
    )
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/api/v1",
        help="API base url (default: http://127.0.0.1:8000/api/v1).",
    )
    p.add_argument(
        "--ground-truth",
        default="data/ground_truth/retrieval_user2/retrieval_ground_truth_final_qdesign.json",
        help="Path to ground-truth JSON.",
    )
    p.add_argument(
        "--srs-dir",
        default="data/srs",
        help="Directory containing source SRS files referenced by ground truth.",
    )
    p.add_argument("--k", type=int, default=5, help="Retrieval depth k.")
    p.add_argument(
        "--mode",
        choices=["unified_project", "project_per_file"],
        default="unified_project",
        help=(
            "Evaluation mode: unified_project uploads all docs into one project (recommended for file-level metrics), "
            "project_per_file creates one project per file."
        ),
    )
    p.add_argument(
        "--keep-projects",
        action="store_true",
        help="Keep temporary projects (default: delete after evaluation).",
    )
    p.add_argument(
        "--output",
        default="results/retrieval_api_eval/report.json",
        help="Output report path.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds for API requests.",
    )
    return p.parse_args()


def load_ground_truth(path: Path) -> list[dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Ground truth must be a JSON list.")
    required = {"question_id", "question", "srs_file"}
    for idx, row in enumerate(items):
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"Ground truth row {idx} missing keys: {sorted(missing)}")
    return items


def create_project(session: requests.Session, base_url: str, name: str, timeout: int) -> str:
    payload = {"name": name, "description": "temporary retrieval evaluation project"}
    r = session.post(f"{base_url}/projects/", json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    project_id = body.get("id")
    if not project_id:
        raise RuntimeError(f"Project creation response missing id: {body}")
    return str(project_id)


def delete_project(session: requests.Session, base_url: str, project_id: str, timeout: int) -> None:
    r = session.delete(f"{base_url}/projects/{project_id}", timeout=timeout)
    if r.status_code not in (200, 204):
        print(f"[warn] failed deleting project {project_id}: {r.status_code} {r.text[:200]}")


def upload_file(
    session: requests.Session, base_url: str, project_id: str, file_path: Path, timeout: int
) -> dict[str, Any]:
    with file_path.open("rb") as fh:
        files = {"srs": (file_path.name, fh, "application/pdf")}
        r = session.post(
            f"{base_url}/projects/{project_id}/upload-srs",
            files=files,
            timeout=timeout,
        )
    r.raise_for_status()
    return r.json()


def retrieve(
    session: requests.Session, base_url: str, project_id: str, query: str, k: int, timeout: int
) -> list[dict[str, Any]]:
    payload = {"query": query, "k": k}
    r = session.post(
        f"{base_url}/projects/{project_id}/retrieval",
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    body = r.json()
    return body.get("results", []) or []


def unique_ranked_files(results: list[dict[str, Any]]) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for item in results:
        meta = item.get("metadata") or {}
        source_file = meta.get("source_file")
        if not source_file or source_file in seen:
            continue
        seen.add(source_file)
        ranked.append(str(source_file))
    return ranked


def score_one(relevant_file: str, ranked_files: list[str], k: int) -> QueryResult:
    topk = ranked_files[:k]
    rel_in_topk = 1.0 if relevant_file in topk else 0.0

    rr = 0.0
    for i, f in enumerate(ranked_files, start=1):
        if f == relevant_file:
            rr = 1.0 / float(i)
            break

    return QueryResult(
        question_id="",
        question="",
        relevant_file=relevant_file,
        ranked_files=ranked_files,
        hit_at_k=rel_in_topk,
        precision_at_k=rel_in_topk / float(max(1, k)),
        recall_at_k=rel_in_topk,
        reciprocal_rank=rr,
    )


def run_unified_mode(
    session: requests.Session,
    base_url: str,
    gt_rows: list[dict[str, Any]],
    srs_dir: Path,
    k: int,
    timeout: int,
) -> tuple[list[str], list[QueryResult]]:
    projects_to_cleanup: list[str] = []
    stamp = int(time.time())
    project_id = create_project(session, base_url, f"retrieval_eval_unified_{stamp}", timeout)
    projects_to_cleanup.append(project_id)
    print(f"[info] created unified project: {project_id}")

    files_needed = sorted({row["srs_file"] for row in gt_rows})
    indexed_name_by_original: dict[str, str] = {}
    for file_name in files_needed:
        fp = srs_dir / file_name
        if not fp.exists():
            raise FileNotFoundError(f"SRS file not found: {fp}")
        print(f"[info] uploading {file_name} ...")
        upload_res = upload_file(session, base_url, project_id, fp, timeout)
        indexed_name_by_original[file_name] = str(upload_res.get("stored_as") or file_name)

    scored: list[QueryResult] = []
    for row in gt_rows:
        results = retrieve(session, base_url, project_id, row["question"], k, timeout)
        ranked = unique_ranked_files(results)
        relevant_file = indexed_name_by_original.get(str(row["srs_file"]), str(row["srs_file"]))
        q = score_one(relevant_file, ranked, k)
        q.question_id = str(row["question_id"])
        q.question = str(row["question"])
        scored.append(q)
    return projects_to_cleanup, scored


def run_project_per_file_mode(
    session: requests.Session,
    base_url: str,
    gt_rows: list[dict[str, Any]],
    srs_dir: Path,
    k: int,
    timeout: int,
) -> tuple[list[str], list[QueryResult]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in gt_rows:
        grouped.setdefault(str(row["srs_file"]), []).append(row)

    projects_to_cleanup: list[str] = []
    scored: list[QueryResult] = []

    stamp = int(time.time())
    for file_name, rows in grouped.items():
        fp = srs_dir / file_name
        if not fp.exists():
            raise FileNotFoundError(f"SRS file not found: {fp}")

        project_id = create_project(
            session, base_url, f"retrieval_eval_{file_name}_{stamp}".replace(" ", "_"), timeout
        )
        projects_to_cleanup.append(project_id)
        print(f"[info] created project for {file_name}: {project_id}")
        upload_res = upload_file(session, base_url, project_id, fp, timeout)
        indexed_file_name = str(upload_res.get("stored_as") or file_name)

        for row in rows:
            results = retrieve(session, base_url, project_id, row["question"], k, timeout)
            ranked = unique_ranked_files(results)
            q = score_one(indexed_file_name, ranked, k)
            q.question_id = str(row["question_id"])
            q.question = str(row["question"])
            scored.append(q)

    return projects_to_cleanup, scored


def aggregate(rows: list[QueryResult]) -> dict[str, float]:
    n = float(max(1, len(rows)))
    return {
        "queries": len(rows),
        "hit@k": sum(r.hit_at_k for r in rows) / n,
        "precision@k": sum(r.precision_at_k for r in rows) / n,
        "recall@k": sum(r.recall_at_k for r in rows) / n,
        "mrr": sum(r.reciprocal_rank for r in rows) / n,
    }


def main() -> None:
    args = parse_args()
    gt_path = Path(args.ground_truth)
    srs_dir = Path(args.srs_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gt_rows = load_ground_truth(gt_path)
    session = requests.Session()
    projects_to_cleanup: list[str] = []

    try:
        if args.mode == "unified_project":
            projects_to_cleanup, scored = run_unified_mode(
                session, args.base_url, gt_rows, srs_dir, args.k, args.timeout
            )
        else:
            projects_to_cleanup, scored = run_project_per_file_mode(
                session, args.base_url, gt_rows, srs_dir, args.k, args.timeout
            )

        summary = aggregate(scored)
        report = {
            "config": {
                "base_url": args.base_url,
                "ground_truth": str(gt_path),
                "srs_dir": str(srs_dir),
                "k": args.k,
                "mode": args.mode,
            },
            "summary": summary,
            "per_query": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "relevant_file": r.relevant_file,
                    "ranked_files": r.ranked_files,
                    "hit@k": r.hit_at_k,
                    "precision@k": r.precision_at_k,
                    "recall@k": r.recall_at_k,
                    "rr": r.reciprocal_rank,
                }
                for r in scored
            ],
        }
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n=== Retrieval API Evaluation ===")
        print(f"mode           : {args.mode}")
        print(f"queries        : {summary['queries']}")
        print(f"hit@{args.k}       : {summary['hit@k']:.4f}")
        print(f"precision@{args.k} : {summary['precision@k']:.4f}")
        print(f"recall@{args.k}    : {summary['recall@k']:.4f}")
        print(f"mrr            : {summary['mrr']:.4f}")
        print(f"report         : {output_path}")
    finally:
        if args.keep_projects:
            print("[info] keeping temporary projects.")
        else:
            for pid in projects_to_cleanup:
                delete_project(session, args.base_url, pid, args.timeout)


if __name__ == "__main__":
    main()
