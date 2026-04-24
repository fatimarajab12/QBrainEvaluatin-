from __future__ import annotations

from typing import Any

from api.services.supabase_client import get_supabase_client
from api.services.supabase_query import safe_execute


class VectorRepository:
    def insert_chunk(self, row: dict[str, Any]) -> None:
        safe_execute("project_vectors.insert", lambda: get_supabase_client().table("project_vectors").insert(row).execute())

    def delete_by_metadata(self, *, project_id: str, metadata_filter: dict[str, Any]) -> None:
        safe_execute(
            "project_vectors.delete_by_metadata",
            lambda: get_supabase_client()
            .table("project_vectors")
            .delete()
            .eq("project_id", project_id)
            .contains("metadata", metadata_filter)
            .execute(),
        )


vector_repository = VectorRepository()
