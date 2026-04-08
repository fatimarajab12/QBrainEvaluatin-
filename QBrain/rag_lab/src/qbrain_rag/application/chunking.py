"""Text splitting (application-level: chunking policy)."""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from qbrain_rag.config.settings import get_settings


def chunk_text(text: str) -> list[str]:
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
    )
    return splitter.split_text(text)
