from __future__ import annotations

from pydantic import BaseModel


class ProjectCreateRequest(BaseModel):
    name: str
    doc_path: str | None = None
    description: str | None = None


class ProjectUpdateRequest(BaseModel):
    name: str | None = None
    doc_path: str | None = None
    description: str | None = None


class ProjectRetrievalRequest(BaseModel):
    query: str
    k: int = 5
