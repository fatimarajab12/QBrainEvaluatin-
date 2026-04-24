from __future__ import annotations

from typing import Any

from api.services.supabase_client import get_supabase_client
from api.services.supabase_query import safe_execute


class PerformanceMetricsRepository:
    def insert(self, *, project_id: str, metric_type: str, payload: dict[str, Any]) -> None:
        safe_execute(
            "performance_metrics.insert",
            lambda: get_supabase_client()
            .table("performance_metrics")
            .insert({"project_id": project_id, "metric_type": metric_type, "payload": payload})
            .execute(),
        )

    def list_recent(self, *, project_id: str, metric_type: str, limit: int):
        return safe_execute(
            "performance_metrics.list_recent",
            lambda: get_supabase_client()
            .table("performance_metrics")
            .select("*")
            .eq("project_id", project_id)
            .eq("metric_type", metric_type)
            .order("created_at", desc=True)
            .limit(limit)
            .execute(),
        )


performance_metrics_repository = PerformanceMetricsRepository()
