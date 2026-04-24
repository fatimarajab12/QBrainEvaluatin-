from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, File, HTTPException, UploadFile

from api.routes.schemes.features import FeatureGenerateRequest
from api.routes.schemes.projects import ProjectCreateRequest, ProjectRetrievalRequest, ProjectUpdateRequest
from api.services.feature_service import feature_service
from api.services.project_service import project_service

projects_router = APIRouter(prefix="/api/v1/projects", tags=["api_v1", "projects"])


@projects_router.post("/")
def create_project(payload: ProjectCreateRequest) -> dict[str, Any]:
    return project_service.create_project(payload.model_dump())


@projects_router.get("/")
def get_user_projects() -> list[dict[str, Any]]:
    return project_service.get_user_projects()


@projects_router.post("/{project_id}/upload-srs")
def upload_srs(
    project_id: str,
    srs: UploadFile | None = File(default=None),
    file: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    upload = srs or file
    if upload is None:
        raise HTTPException(status_code=400, detail="No file uploaded. Use form-data key 'srs' or 'file'.")
    return project_service.upload_srs(project_id, upload)


@projects_router.post("/{project_id}/extract-features-from-doc")
def extract_features_from_processed_doc(
    project_id: str,
    payload: FeatureGenerateRequest | None = Body(default=None),
) -> dict[str, Any]:
    """Re-run RAG document pipeline on ``doc_path`` and persist features (and test cases if ``skip_tests`` is false)."""
    body = payload or FeatureGenerateRequest()
    return feature_service.generate_features_from_srs(project_id, body.model_dump(exclude_none=True))


@projects_router.post("/{project_id}/retrieval")
def project_retrieval(project_id: str, payload: ProjectRetrievalRequest) -> dict[str, Any]:
    return project_service.retrieval(project_id, payload.query, payload.k)


@projects_router.get("/{project_id}/stats")
def get_project_stats(project_id: str) -> dict[str, Any]:
    return project_service.get_project_stats(project_id)


@projects_router.get("/{project_id}/test-cases-count")
def get_project_test_cases_count(project_id: str) -> dict[str, Any]:
    return project_service.get_test_cases_count(project_id)


@projects_router.get("/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    project = project_service.get_project_by_id(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@projects_router.put("/{project_id}")
def update_project(project_id: str, payload: ProjectUpdateRequest) -> dict[str, Any]:
    return project_service.update_project(project_id, payload.model_dump(exclude_none=True))


@projects_router.delete("/{project_id}")
def delete_project(project_id: str) -> dict[str, bool]:
    result = project_service.delete_project(project_id)
    return {"deleted": bool(result.get("success"))}
