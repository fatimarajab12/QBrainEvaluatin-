from __future__ import annotations

from pydantic import BaseModel


class TestCaseCreateRequest(BaseModel):
    project_id: str
    feature_id: str
    title: str
    steps: list[str]
    expected_result: str


class TestCaseUpdateRequest(BaseModel):
    title: str | None = None
    steps: list[str] | None = None
    expected_result: str | None = None


class TestCaseGenerateRequest(BaseModel):
    doc_path: str | None = None
    test_context_k: int = 5
