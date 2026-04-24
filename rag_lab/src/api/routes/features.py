from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from api.routes.schemes.features import FeatureCreateRequest, FeatureGenerateRequest, FeatureUpdateRequest
from api.services.feature_service import feature_service

features_router = APIRouter(prefix="/api/v1/features", tags=["api_v1", "features"])


@features_router.get("/projects/{project_id}/features")
def get_project_features(project_id: str) -> list[dict[str, Any]]:
    return feature_service.get_project_features(project_id)


@features_router.get("/projects/{project_id}/has-ai-features")
def check_has_ai_generated_features(project_id: str) -> dict[str, Any]:
    return {"project_id": project_id, "has_ai_features": feature_service.has_ai_generated_features(project_id)}


@features_router.get("/projects/{project_id}/performance-metrics")
def get_performance_metrics(project_id: str) -> dict[str, Any]:
    return feature_service.get_performance_metrics(project_id)


@features_router.post("/projects/{project_id}/generate-features")
def generate_features(project_id: str, payload: FeatureGenerateRequest) -> dict[str, Any]:
    return feature_service.generate_features_from_srs(project_id, payload.model_dump(exclude_none=True))


@features_router.post("/bulk")
def bulk_create_features(payload: list[FeatureCreateRequest]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in payload:
        grouped.setdefault(item.project_id, []).append(item.model_dump())
    created: list[dict[str, Any]] = []
    for project_id, items in grouped.items():
        created.extend(feature_service.bulk_create_features(project_id, items))
    return {"count": len(created), "items": created}


@features_router.get("/{feature_id}/test-cases-count")
def get_test_cases_count(feature_id: str) -> dict[str, Any]:
    return feature_service.get_test_cases_count(feature_id)


@features_router.get("/{feature_id}")
def get_feature(feature_id: str) -> dict[str, Any]:
    feature = feature_service.get_feature_by_id(feature_id)
    if not feature:
        raise HTTPException(status_code=404, detail="Feature not found")
    return feature


@features_router.post("/")
def create_feature(payload: FeatureCreateRequest) -> dict[str, Any]:
    return feature_service.create_feature(payload.model_dump())


@features_router.put("/{feature_id}")
def update_feature(feature_id: str, payload: FeatureUpdateRequest) -> dict[str, Any]:
    return feature_service.update_feature(feature_id, payload.model_dump(exclude_none=True))


@features_router.delete("/{feature_id}")
def delete_feature(feature_id: str) -> dict[str, bool]:
    result = feature_service.delete_feature(feature_id)
    return {"deleted": bool(result.get("success"))}
