from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from api.controllers.rag_controller import RAGController
from api.routes.schemes.rag import (
    DocumentPipelineRequest,
    IngestionRequest,
    QueryRequest,
    RetrievalRequest,
)

rag_router = APIRouter(prefix="/api/v1/rag", tags=["api_v1", "rag"])
rag_controller = RAGController()


@rag_router.post("/ingestion")
def ingestion(payload: IngestionRequest) -> dict[str, Any]:
    return rag_controller.ingestion(payload.doc_path)


@rag_router.post("/retrieval")
def retrieval(payload: RetrievalRequest) -> dict[str, Any]:
    return rag_controller.retrieval(payload.doc_path, payload.query, payload.k)


@rag_router.post("/query")
def query(payload: QueryRequest) -> dict[str, Any]:
    return rag_controller.query(payload.doc_path, payload.question, payload.k, payload.temperature)


@rag_router.post("/document-pipeline")
def document_pipeline(payload: DocumentPipelineRequest) -> dict[str, Any]:
    return rag_controller.document_pipeline(
        payload.doc_path,
        payload.test_context_k,
        payload.max_features,
        payload.skip_tests,
        payload.quiet,
        payload.project_id,
    )
