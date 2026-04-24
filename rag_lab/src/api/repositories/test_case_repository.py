from __future__ import annotations

from typing import Any

from api.services.supabase_client import get_supabase_client
from api.services.supabase_query import safe_execute


class TestCaseRepository:
    def insert(self, payload: dict[str, Any]):
        return safe_execute("test_cases.insert", lambda: get_supabase_client().table("test_cases").insert(payload).execute())

    def fetch_by_id(self, test_case_id: str):
        return safe_execute(
            "test_cases.fetch_by_id",
            lambda: get_supabase_client().table("test_cases").select("*").eq("id", test_case_id).limit(1).execute(),
        )

    def list_by_feature(self, feature_id: str):
        return safe_execute(
            "test_cases.list_by_feature",
            lambda: get_supabase_client()
            .table("test_cases")
            .select("*")
            .eq("feature_id", feature_id)
            .order("created_at", desc=True)
            .execute(),
        )

    def list_by_project(self, project_id: str):
        return safe_execute(
            "test_cases.list_by_project",
            lambda: get_supabase_client()
            .table("test_cases")
            .select("*")
            .eq("project_id", project_id)
            .order("created_at", desc=True)
            .execute(),
        )

    def list_features_by_project(self, project_id: str):
        return safe_execute(
            "features.list_for_grouping",
            lambda: get_supabase_client().table("features").select("*").eq("project_id", project_id).execute(),
        )

    def update(self, test_case_id: str, update_data: dict[str, Any]):
        return safe_execute(
            "test_cases.update",
            lambda: get_supabase_client().table("test_cases").update(update_data).eq("id", test_case_id).execute(),
        )

    def delete(self, test_case_id: str):
        return safe_execute("test_cases.delete", lambda: get_supabase_client().table("test_cases").delete().eq("id", test_case_id).execute())

    def fetch_feature_row(self, feature_id: str):
        return safe_execute(
            "features.fetch_for_test_generation",
            lambda: get_supabase_client().table("features").select("*").eq("id", feature_id).limit(1).execute(),
        )

    def fetch_project_row(self, project_id: str):
        return safe_execute(
            "projects.fetch_for_test_generation",
            lambda: get_supabase_client().table("projects").select("*").eq("id", project_id).limit(1).execute(),
        )

    def count_ai_by_feature(self, feature_id: str):
        return safe_execute(
            "test_cases.count_ai_by_feature",
            lambda: get_supabase_client()
            .table("test_cases")
            .select("id", count="exact")
            .eq("feature_id", feature_id)
            .eq("is_ai_generated", True)
            .execute(),
        )


test_case_repository = TestCaseRepository()
