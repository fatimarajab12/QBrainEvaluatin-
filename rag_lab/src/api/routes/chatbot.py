from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from api.controllers.rag_controller import RAGController
from api.routes.schemes.chatbot import ChatbotContextRequest, ChatbotQueryRequest

chatbot_router = APIRouter(prefix="/api/v1/chatbot", tags=["api_v1", "chatbot"])
rag_controller = RAGController()


@chatbot_router.post("/query")
def query_chatbot(payload: ChatbotQueryRequest) -> dict[str, Any]:
    return rag_controller.query(
        doc_path=payload.doc_path,
        question=payload.question,
        k=payload.k,
        temperature=payload.temperature,
    )


@chatbot_router.post("/context")
def get_chatbot_context(payload: ChatbotContextRequest) -> dict[str, Any]:
    return rag_controller.retrieval(
        doc_path=payload.doc_path,
        query=payload.query,
        k=payload.k,
    )
