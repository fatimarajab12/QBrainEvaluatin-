from __future__ import annotations

from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    doc_path: str = Field(..., description="Absolute or relative path to a source document")


class RetrievalRequest(BaseModel):
    doc_path: str = Field(..., description="Absolute or relative path to a source document")
    query: str
    k: int = 5


class QueryRequest(BaseModel):
    doc_path: str = Field(..., description="Absolute or relative path to a source document")
    question: str
    k: int = 5
    temperature: float | None = None


class DocumentPipelineRequest(BaseModel):
    doc_path: str = Field(..., description="Absolute or relative path to a source document")
    test_context_k: int = 5
    max_features: int | None = None
    skip_tests: bool = False
    quiet: bool = False
    project_id: str | None = Field(
        default=None,
        description="When set, vectors are written under this Supabase project id (UUID).",
    )
