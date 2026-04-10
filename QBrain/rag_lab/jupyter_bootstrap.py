"""Jupyter / notebook entry: run via runpy.run_path() so __file__ resolves correctly."""
from __future__ import annotations

import sys
from pathlib import Path

RAG_LAB = Path(__file__).resolve().parent
_SRC = RAG_LAB / "src"
if not (_SRC / "qbrain_rag").is_dir():
    raise RuntimeError(f"jupyter_bootstrap.py must live in rag_lab root; got {RAG_LAB}")
_p = str(_SRC)
if _p not in sys.path:
    sys.path.insert(0, _p)
