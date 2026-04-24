from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from application.chunking import chunk_text
from infrastructure.document_loaders import load_document
from services.rag_service import RAGService


class RAGApiService:
    def __init__(self) -> None:
        self._service = RAGService()

    @staticmethod
    def resolve_existing_file(doc_path: str) -> Path:
        path = Path(doc_path).expanduser().resolve()
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        return path

    @staticmethod
    def serialize_docs(docs: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "metadata": getattr(d, "metadata", {}),
                "page_content": getattr(d, "page_content", ""),
            }
            for d in docs
        ]

    def ingestion_summary(self, doc_path: str) -> dict[str, Any]:
        doc = self.resolve_existing_file(doc_path)
        text = load_document(str(doc))
        chunks = chunk_text(text)
        return {
            "file": doc.name,
            "characters": len(text),
            "chunks": len(chunks),
            "first_chunk_preview": chunks[0][:600] if chunks else "",
        }

    def retrieval(self, doc_path: str, query: str, k: int) -> dict[str, Any]:
        doc = self.resolve_existing_file(doc_path)
        store = self._service.build_store_from_path(doc)
        docs = self._service.retrieve(store, query, k=k)
        return {"query": query, "k": k, "results": self.serialize_docs(docs)}

    def query(self, doc_path: str, question: str, k: int, temperature: float | None) -> dict[str, Any]:
        doc = self.resolve_existing_file(doc_path)
        store = self._service.build_store_from_path(doc)
        docs = self._service.retrieve(store, question, k=k)
        answer = self._service.answer(question, docs, temperature=temperature)
        return {
            "question": question,
            "k": k,
            "answer": answer,
            "retrieved_context": self.serialize_docs(docs),
        }

    def document_pipeline(
        self,
        doc_path: str,
        test_context_k: int,
        max_features: int | None,
        skip_tests: bool,
        quiet: bool,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        doc = self.resolve_existing_file(doc_path)
        return self._service.document_features_and_tests(
            doc,
            n_test_context_chunks=test_context_k,
            max_features=max_features,
            skip_test_cases=skip_tests,
            verbose=not quiet,
            project_id=project_id,
        )
