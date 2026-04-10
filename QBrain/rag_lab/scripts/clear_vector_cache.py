"""
Delete on-disk FAISS / vector artifacts under ``rag_lab/``.

The default **document pipeline**, **benchmark**, and **notebooks** call ``build_faiss_store`` and keep the
index **only in memory**. Persisted data appears only if you used ``save_faiss_store``, a custom path, or
``verify_rag_index.py --save-cache``.

Removes:

- ``data/faiss_cache/`` (conventional cache; in ``.gitignore``)
- Any directory under ``rag_lab/`` (except ``.venv`` / ``.git``) that contains both ``index.faiss`` and ``index.pkl``

Usage (from ``rag_lab/``):

  python scripts/clear_vector_cache.py
  python scripts/clear_vector_cache.py --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_SKIP_PARTS = frozenset({".git", ".venv", "__pycache__", ".ipynb_checkpoints", "node_modules"})


def _skip_path(p: Path) -> bool:
    return not _SKIP_PARTS.isdisjoint(p.parts)


def main() -> None:
    p = argparse.ArgumentParser(description="Remove persisted FAISS / vector cache under rag_lab.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    removed: list[str] = []

    faiss_cache = ROOT / "data" / "faiss_cache"
    if faiss_cache.exists():
        if args.dry_run:
            print(f"[dry-run] would remove: {faiss_cache}")
        else:
            shutil.rmtree(faiss_cache, ignore_errors=True)
            removed.append(str(faiss_cache))

    seen: set[Path] = set()
    for faiss_file in ROOT.rglob("index.faiss"):
        parent = faiss_file.parent.resolve()
        try:
            rel = parent.relative_to(ROOT.resolve())
        except ValueError:
            continue
        if _skip_path(rel):
            continue
        if not (parent / "index.pkl").is_file():
            continue
        if parent in seen:
            continue
        seen.add(parent)
        if args.dry_run:
            print(f"[dry-run] would remove FAISS dir: {parent}")
        else:
            shutil.rmtree(parent, ignore_errors=True)
            removed.append(str(parent))

    if args.dry_run:
        if not seen and not faiss_cache.exists():
            print("Nothing to remove (dry-run).")
        return

    if removed:
        print("Removed:")
        for r in removed:
            print(" ", r)
    else:
        print("Nothing to remove: no data/faiss_cache and no LangChain FAISS folders (index.faiss + index.pkl).")
    print(
        "Note: the usual pipeline rebuilds FAISS in RAM each run; there is no separate vector DB file to wipe."
    )


if __name__ == "__main__":
    main()
