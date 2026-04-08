"""Default SRS path for Jupyter notebooks (JDECo lab PDF)."""
from __future__ import annotations

from pathlib import Path

# Basename expected in repo or next to workspace root
DEFAULT_SRS_BASENAME = "JDECo_SRS.docx[1].pdf"


def resolve_default_srs_path(rag_lab: Path) -> Path:
    """
    Resolve the real JDECo SRS PDF used across notebooks.

    Search order:
    1. ``<parent-of-QBrain>/JDECo_SRS.docx[1].pdf`` (e.g. ``D:/Qbrainpython/``)
    2. ``rag_lab/data/srs/JDECo_SRS.docx[1].pdf``
    """
    root = rag_lab.resolve()
    candidates = [
        root.parent.parent / DEFAULT_SRS_BASENAME,
        root / "data" / "srs" / DEFAULT_SRS_BASENAME,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Could not find {DEFAULT_SRS_BASENAME!r}. "
        f"Place it next to your workspace (e.g. alongside QBrain/) or under {root / 'data' / 'srs'}."
    )
