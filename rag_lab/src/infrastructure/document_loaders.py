"""File ingestion: PDF, plain text, and simple HTML (e.g. NL Requirements dataset)."""
from __future__ import annotations

import re
from pathlib import Path

import fitz  # pymupdf


def load_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)


def load_txt_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_html_text(path: str) -> str:
    """Best-effort text extraction from HTML/HTM (no extra dependencies)."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    raw = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def load_document(path: str) -> str:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf_text(str(path))
    if suffix == ".txt":
        return load_txt_text(str(path))
    if suffix in (".html", ".htm"):
        return load_html_text(str(path))

    raise ValueError(
        "Unsupported file type. Use .pdf, .txt, .html, or .htm. "
        "For .doc/.docx/.rtf from external datasets, export to PDF or plain text first."
    )
