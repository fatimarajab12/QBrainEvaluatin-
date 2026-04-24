from __future__ import annotations

import shutil
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from pathlib import Path

from api.repositories.project_repository import project_repository
from api.services.supabase_query import parse_uuid
from application.chunking import chunk_text
from config.settings import get_settings
from infrastructure.document_loaders import load_document
from infrastructure.vector_store import SupabaseStore, build_vector_store, retrieve_top_k


class ProjectService:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._projects = project_repository

    def create_project(self, project_data: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "name": project_data["name"],
            "doc_path": project_data.get("doc_path"),
            "description": project_data.get("description"),
        }
        result = self._projects.insert(payload)
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create project in Supabase")
        row = result.data[0]
        return {
            "id": str(row.get("id")),
            "name": row.get("name"),
            "doc_path": row.get("doc_path"),
            "description": row.get("description"),
        }

    def get_project_by_id(self, project_id: str) -> dict[str, Any] | None:
        project_id = parse_uuid(project_id, field="project id")
        result = self._projects.fetch_by_id(project_id)
        if not result.data:
            return None
        row = result.data[0]
        return {
            "id": str(row.get("id")),
            "name": row.get("name"),
            "doc_path": row.get("doc_path"),
            "description": row.get("description"),
        }

    def get_user_projects(self) -> list[dict[str, Any]]:
        result = self._projects.list_all()
        return [
            {
                "id": str(row.get("id")),
                "name": row.get("name"),
                "doc_path": row.get("doc_path"),
                "description": row.get("description"),
            }
            for row in (result.data or [])
        ]

    def update_project(self, project_id: str, update_data: dict[str, Any]) -> dict[str, Any]:
        project_id = parse_uuid(project_id, field="project id")
        result = self._projects.update(project_id, update_data)
        if not result.data:
            raise HTTPException(status_code=404, detail="Project not found")
        row = result.data[0]
        return {
            "id": str(row.get("id")),
            "name": row.get("name"),
            "doc_path": row.get("doc_path"),
            "description": row.get("description"),
        }

    def delete_project(self, project_id: str) -> dict[str, Any]:
        project_id = parse_uuid(project_id, field="project id")
        self._projects.delete_vectors_for_project(project_id)
        self._projects.delete_test_cases_for_project(project_id)
        self._projects.delete_features_for_project(project_id)
        result = self._projects.delete(project_id)
        if not result.data:
            raise HTTPException(status_code=404, detail="Project not found")
        return {"success": True}

    def get_project_stats(self, project_id: str) -> dict[str, Any]:
        project_id = parse_uuid(project_id, field="project id")
        project = self.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        features_resp = self._projects.count_features(project_id)
        test_cases_resp = self._projects.count_test_cases(project_id)
        features_count = int(features_resp.count or 0)
        test_cases_count = int(test_cases_resp.count or 0)
        return {"projectId": project_id, "name": project["name"], "featuresCount": int(features_count), "testCasesCount": int(test_cases_count)}

    def get_test_cases_count(self, project_id: str) -> dict[str, Any]:
        project_id = parse_uuid(project_id, field="project id")
        project = self.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        test_cases_resp = self._projects.count_test_cases(project_id)
        test_cases_count = int(test_cases_resp.count or 0)
        return {"projectId": project_id, "testCasesCount": int(test_cases_count)}

    def retrieval(self, project_id: str, query: str, k: int = 5) -> dict[str, Any]:
        project_id = parse_uuid(project_id, field="project id")
        project = self.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        store = SupabaseStore(project_id=project_id, docs=[], vectors=[])
        docs = retrieve_top_k(store, query, k=max(1, int(k)))
        return {
            "projectId": project_id,
            "query": query,
            "k": int(k),
            "results": [{"metadata": d.metadata, "page_content": d.page_content} for d in docs],
        }

    def upload_srs(self, project_id: str, file: UploadFile) -> dict[str, Any]:
        project_id = parse_uuid(project_id, field="project id")
        project = self.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing file name")

        ext = Path(file.filename).suffix.lower()
        if ext not in {".pdf", ".txt"}:
            raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")

        safe_name = f"{project_id}_{uuid4().hex}{ext}"
        rag_lab_root = Path(__file__).resolve().parents[3]
        # Keep generated upload artifacts separate from original SRS dataset files.
        upload_dir = rag_lab_root / "data" / "srs" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        local_path = upload_dir / safe_name
        with local_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        bucket = self._settings.supabase_srs_bucket
        if self._settings.supabase_srs_storage_upload:
            content = local_path.read_bytes()
            storage_path = f"uploads/{safe_name}"
            self._projects.upload_object(
                bucket=bucket,
                path=storage_path,
                data=content,
                content_type=file.content_type or "application/octet-stream",
            )
            stored_path = f"{bucket}/{storage_path}"
        else:
            stored_path = str(local_path.resolve())
        self.update_project(project_id, {"doc_path": str(local_path.resolve())})

        # Node-style behavior: upload endpoint prepares only the knowledge source.
        text = load_document(str(local_path.resolve()))
        chunks = chunk_text(text)
        metadata = [
            {"source_file": local_path.name, "chunk_id": i + 1, "project_id": project_id}
            for i in range(len(chunks))
        ]
        build_vector_store(chunks, metadata)

        return {
            "projectId": project_id,
            "filename": file.filename,
            "stored_as": safe_name,
            "doc_path": stored_path,
            "processing": {
                "chunk_count": len(chunks),
                "embeddings_indexed": len(chunks),
                "knowledge_source_ready": True,
                "feature_count": 0,
                "test_case_count": 0,
            },
        }


project_service = ProjectService()
