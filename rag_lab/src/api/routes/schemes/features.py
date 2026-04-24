from __future__ import annotations

from pydantic import BaseModel


class FeatureCreateRequest(BaseModel):
    project_id: str
    title: str
    content: str | None = None


class FeatureUpdateRequest(BaseModel):
    title: str | None = None
    content: str | None = None


class FeatureGenerateRequest(BaseModel):
    doc_path: str | None = None
    skip_tests: bool = True
    test_context_k: int = 5
