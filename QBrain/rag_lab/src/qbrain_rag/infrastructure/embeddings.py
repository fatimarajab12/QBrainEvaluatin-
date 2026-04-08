"""OpenAI embeddings adapter (LangChain)."""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from qbrain_rag.config.settings import get_settings


def get_embedding_model() -> OpenAIEmbeddings:
    s = get_settings()
    if not s.openai_api_key:
        raise ValueError("Set OPENAI_API_KEY in rag_lab/.env")
    return OpenAIEmbeddings(model=s.embedding_model, openai_api_key=s.openai_api_key)
