"""Central configuration loaded from environment (`.env` in `rag_lab/`)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Project root: .../rag_lab
_RAG_LAB_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_RAG_LAB_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    embedding_model: str
    chat_model: str
    chunk_size: int
    chunk_overlap: int
    default_top_k: int
    generation_temperature: float


@lru_cache
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        embedding_model="text-embedding-3-small",
        chat_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        chunk_size=2000,
        chunk_overlap=300,
        default_top_k=int(os.getenv("RAG_TOP_K", "5")),
        generation_temperature=float(os.getenv("GENERATION_TEMPERATURE", "0.7")),
    )
