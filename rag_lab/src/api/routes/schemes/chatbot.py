from __future__ import annotations

from pydantic import BaseModel


class ChatbotQueryRequest(BaseModel):
    doc_path: str
    question: str
    k: int = 5
    temperature: float | None = None


class ChatbotContextRequest(BaseModel):
    doc_path: str
    query: str
    k: int = 5
