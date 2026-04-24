from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from api.routes.schemes.test_cases import TestCaseCreateRequest, TestCaseGenerateRequest, TestCaseUpdateRequest
from api.services.test_case_service import test_case_service

test_cases_router = APIRouter(prefix="/api/v1/test-cases", tags=["api_v1", "test-cases"])


@test_cases_router.get("/features/{feature_id}/test-cases")
def get_feature_test_cases(feature_id: str) -> list[dict[str, Any]]:
    return test_case_service.get_feature_test_cases(feature_id)


@test_cases_router.get("/features/{feature_id}/has-ai-test-cases")
def check_has_ai_generated_test_cases(feature_id: str) -> dict[str, Any]:
    return {"feature_id": feature_id, "has_ai_test_cases": test_case_service.has_ai_generated_test_cases(feature_id)}


@test_cases_router.get("/feature/{feature_id}")
def get_feature_test_cases_short(feature_id: str) -> list[dict[str, Any]]:
    return test_case_service.get_feature_test_cases(feature_id)


@test_cases_router.get("/projects/{project_id}/test-cases")
def get_project_test_cases(project_id: str) -> list[dict[str, Any]]:
    return test_case_service.get_project_test_cases(project_id)


@test_cases_router.post("/features/{feature_id}/generate-test-cases")
def generate_test_cases(feature_id: str, payload: TestCaseGenerateRequest) -> dict[str, Any]:
    created = test_case_service.generate_test_cases_for_feature(feature_id, payload.model_dump(exclude_none=True))
    return {"generated_count": len(created), "items": created}


@test_cases_router.post("/features/{feature_id}")
def create_test_case_for_feature(feature_id: str, payload: TestCaseCreateRequest) -> dict[str, Any]:
    model = payload.model_dump()
    model["feature_id"] = feature_id
    return test_case_service.create_test_case(model)


@test_cases_router.post("/bulk")
def bulk_create_test_cases(payload: list[TestCaseCreateRequest]) -> dict[str, Any]:
    created: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in payload:
        grouped.setdefault(item.feature_id, []).append(item.model_dump())
    for feature_id, items in grouped.items():
        created.extend(test_case_service.bulk_create_test_cases(feature_id, items))
    return {"count": len(created), "items": created}


@test_cases_router.get("/by-feature/{feature_id}")
def get_test_cases_by_feature(feature_id: str) -> list[dict[str, Any]]:
    return test_case_service.get_feature_test_cases(feature_id)


@test_cases_router.get("/{test_case_id}/check")
def check_test_case_exists(test_case_id: str) -> dict[str, Any]:
    return {"id": test_case_id, "exists": test_case_service.get_test_case_by_id(test_case_id) is not None}


@test_cases_router.get("/{test_case_id}")
def get_test_case(test_case_id: str) -> dict[str, Any]:
    test_case = test_case_service.get_test_case_by_id(test_case_id)
    if not test_case:
        raise HTTPException(status_code=404, detail="Test case not found")
    return test_case


@test_cases_router.post("/")
def create_test_case(payload: TestCaseCreateRequest) -> dict[str, Any]:
    return test_case_service.create_test_case(payload.model_dump())


@test_cases_router.put("/{test_case_id}")
def update_test_case(test_case_id: str, payload: TestCaseUpdateRequest) -> dict[str, Any]:
    return test_case_service.update_test_case(test_case_id, payload.model_dump(exclude_none=True))


@test_cases_router.delete("/{test_case_id}")
def delete_test_case(test_case_id: str) -> dict[str, bool]:
    result = test_case_service.delete_test_case(test_case_id)
    return {"deleted": bool(result.get("success"))}


@test_cases_router.get("/{test_case_id}/gherkin")
def convert_to_gherkin(test_case_id: str) -> dict[str, Any]:
    return test_case_service.convert_test_case_to_gherkin(test_case_id)


@test_cases_router.get("/features/{feature_id}/gherkin")
def convert_feature_test_cases_to_gherkin(feature_id: str) -> dict[str, Any]:
    items = test_case_service.get_feature_test_cases(feature_id)
    return {"feature_id": feature_id, "items": items}


@test_cases_router.get("/projects/{project_id}/all")
def get_all_test_cases_by_features(project_id: str) -> dict[str, Any]:
    return test_case_service.get_all_test_cases_by_features(project_id)


@test_cases_router.get("/projects/{project_id}/export/excel")
def export_all_test_cases_to_excel(project_id: str) -> dict[str, Any]:
    # File export is intentionally disabled; return JSON only.
    items = test_case_service.get_project_test_cases(project_id)
    return {"project_id": project_id, "export": "json_only", "items": items}
