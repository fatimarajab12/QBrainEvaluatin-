from __future__ import annotations

from typing import Any

from api.services.supabase_client import get_supabase_client
from api.services.supabase_query import safe_execute


class ProjectRepository:
    def insert(self, payload: dict[str, Any]):
        return safe_execute("projects.insert", lambda: get_supabase_client().table("projects").insert(payload).execute())

    def fetch_by_id(self, project_id: str):
        return safe_execute(
            "projects.fetch_by_id",
            lambda: get_supabase_client().table("projects").select("*").eq("id", project_id).limit(1).execute(),
        )

    def list_all(self):
        return safe_execute(
            "projects.list_all",
            lambda: get_supabase_client().table("projects").select("*").order("created_at", desc=True).execute(),
        )

    def update(self, project_id: str, update_data: dict[str, Any]):
        return safe_execute(
            "projects.update",
            lambda: get_supabase_client().table("projects").update(update_data).eq("id", project_id).execute(),
        )

    def delete_vectors_for_project(self, project_id: str) -> None:
        safe_execute(
            "project_vectors.delete_by_project",
            lambda: get_supabase_client().table("project_vectors").delete().eq("project_id", project_id).execute(),
        )

    def delete_test_cases_for_project(self, project_id: str) -> None:
        safe_execute(
            "test_cases.delete_by_project",
            lambda: get_supabase_client().table("test_cases").delete().eq("project_id", project_id).execute(),
        )

    def delete_features_for_project(self, project_id: str) -> None:
        safe_execute(
            "features.delete_by_project",
            lambda: get_supabase_client().table("features").delete().eq("project_id", project_id).execute(),
        )

    def delete(self, project_id: str):
        return safe_execute("projects.delete", lambda: get_supabase_client().table("projects").delete().eq("id", project_id).execute())

    def count_features(self, project_id: str):
        return safe_execute(
            "features.count_by_project",
            lambda: get_supabase_client().table("features").select("id", count="exact").eq("project_id", project_id).execute(),
        )

    def count_test_cases(self, project_id: str):
        return safe_execute(
            "test_cases.count_by_project",
            lambda: get_supabase_client().table("test_cases").select("id", count="exact").eq("project_id", project_id).execute(),
        )

    def upload_object(self, *, bucket: str, path: str, data: bytes, content_type: str) -> None:
        client = get_supabase_client()
        safe_execute(
            "storage.upload_object",
            lambda: client.storage.from_(bucket).upload(
                path=path,
                file=data,
                file_options={"content-type": content_type, "upsert": "true"},
            ),
        )


project_repository = ProjectRepository()
