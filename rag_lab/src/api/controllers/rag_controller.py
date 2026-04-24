from __future__ import annotations

from typing import Any

from api.services.rag_api_service import RAGApiService


class RAGController:
    def __init__(self) -> None:
        self._api_service = RAGApiService()

    def ingestion(self, doc_path: str) -> dict[str, Any]:
        return self._api_service.ingestion_summary(doc_path)

    def retrieval(self, doc_path: str, query: str, k: int) -> dict[str, Any]:
        return self._api_service.retrieval(doc_path, query, k)

    def query(self, doc_path: str, question: str, k: int, temperature: float | None) -> dict[str, Any]:
        return self._api_service.query(doc_path, question, k, temperature)

    def document_pipeline(
        self,
        doc_path: str,
        test_context_k: int,
        max_features: int | None,
        skip_tests: bool,
        quiet: bool,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        return self._api_service.document_pipeline(
            doc_path,
            test_context_k,
            max_features,
            skip_tests,
            quiet,
            project_id,
        )
