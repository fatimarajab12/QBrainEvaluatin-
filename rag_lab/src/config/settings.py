"""Central configuration loaded from environment (`.env` in `rag_lab/`)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Project root: .../rag_lab
_RAG_LAB_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_RAG_LAB_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    supabase_url: str | None
    supabase_service_role_key: str | None
    supabase_srs_bucket: str
    # When True, SRS uploads are mirrored to Supabase Storage (requires bucket + policies).
    supabase_srs_storage_upload: bool
    use_supabase: bool
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
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        supabase_srs_bucket=os.getenv("SUPABASE_SRS_BUCKET", "srs-files"),
        supabase_srs_storage_upload=os.getenv("SUPABASE_SRS_STORAGE_UPLOAD", "false").strip().lower()
        in {"1", "true", "yes", "on"},
        use_supabase=os.getenv("USE_SUPABASE", "false").strip().lower() in {"1", "true", "yes", "on"},
        embedding_model="text-embedding-3-small",
        chat_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        chunk_size=2000,
        chunk_overlap=300,
        default_top_k=int(os.getenv("RAG_TOP_K", "5")),
        generation_temperature=float(os.getenv("GENERATION_TEMPERATURE", "0.7")),
    )
