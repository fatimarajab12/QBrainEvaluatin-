"""
Application service: orchestrates ingestion → index → retrieve → generate.

Use this from scripts/notebooks instead of calling infrastructure modules directly
when you want a single entry point (cleaner dependency direction).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from qbrain_rag.application.chunking import chunk_text
from qbrain_rag.infrastructure.vector_store import retrieve_top_k
from qbrain_rag.application.document_pipeline import run_document_pipeline
from qbrain_rag.config.settings import Settings, get_settings
from qbrain_rag.infrastructure.document_loaders import load_document
from qbrain_rag.infrastructure.llm import answer_with_context
from qbrain_rag.infrastructure.vector_store import build_faiss_store

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS


class RAGService:
    """Facade over the mini-lab RAG stack."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def build_store_from_path(self, path: str | Path) -> "FAISS":
        p = Path(path).resolve()
        text = load_document(str(p))
        chunks = chunk_text(text)
        metadata = [{"source_file": p.name, "chunk_id": i + 1} for i in range(len(chunks))]
        return build_faiss_store(chunks, metadata)

    def retrieve(self, store: "FAISS", query: str, k: int | None = None) -> list[Any]:
        return retrieve_top_k(store, query, k=k)

    def answer(self, question: str, docs, *, temperature: float | None = None) -> str:
        return answer_with_context(question, docs, temperature=temperature)

    def query(self, path: str | Path, question: str, k: int | None = None) -> str:
        """One-shot: index file (in memory) → retrieve → generate."""
        store = self.build_store_from_path(path)
        docs = self.retrieve(store, question, k=k)
        return self.answer(question, docs)

    def document_features_and_tests(
        self,
        path: str | Path,
        *,
        n_test_context_chunks: int = 5,
        max_features: int | None = None,
        skip_test_cases: bool = False,
        verbose: bool = True,
    ) -> dict:
        """Index ``path``, extract features (segment merge when long), then test cases per feature (multi-query RAG)."""
        return run_document_pipeline(
            path,
            n_test_context_chunks=n_test_context_chunks,
            max_features=max_features,
            skip_test_cases=skip_test_cases,
            verbose=verbose,
        )
