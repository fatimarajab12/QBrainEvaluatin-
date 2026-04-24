from __future__ import annotations

from typing import Any

from api.services.supabase_client import get_supabase_client
from api.services.supabase_query import safe_execute


class FeatureRepository:
    def insert(self, payload: dict[str, Any]):
        return safe_execute("features.insert", lambda: get_supabase_client().table("features").insert(payload).execute())

    def fetch_by_id(self, feature_id: str):
        return safe_execute(
            "features.fetch_by_id",
            lambda: get_supabase_client().table("features").select("*").eq("id", feature_id).limit(1).execute(),
        )

    def list_by_project(self, project_id: str):
        return safe_execute(
            "features.list_by_project",
            lambda: get_supabase_client()
            .table("features")
            .select("*")
            .eq("project_id", project_id)
            .order("created_at", desc=True)
            .execute(),
        )

    def count_ai_by_project(self, project_id: str):
        return safe_execute(
            "features.count_ai_by_project",
            lambda: get_supabase_client()
            .table("features")
            .select("id", count="exact")
            .eq("project_id", project_id)
            .eq("is_ai_generated", True)
            .execute(),
        )

    def update(self, feature_id: str, update_data: dict[str, Any]):
        return safe_execute(
            "features.update",
            lambda: get_supabase_client().table("features").update(update_data).eq("id", feature_id).execute(),
        )

    def delete(self, feature_id: str):
        return safe_execute("features.delete", lambda: get_supabase_client().table("features").delete().eq("id", feature_id).execute())

    def fetch_project_row(self, project_id: str):
        return safe_execute(
            "projects.fetch_for_feature_generation",
            lambda: get_supabase_client().table("projects").select("*").eq("id", project_id).limit(1).execute(),
        )

    def count_test_cases_by_feature(self, feature_id: str):
        return safe_execute(
            "test_cases.count_by_feature",
            lambda: get_supabase_client().table("test_cases").select("id", count="exact").eq("feature_id", feature_id).execute(),
        )


feature_repository = FeatureRepository()
