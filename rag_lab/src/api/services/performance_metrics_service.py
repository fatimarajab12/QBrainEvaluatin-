from __future__ import annotations

from typing import Any

from api.repositories.performance_metrics_repository import performance_metrics_repository
from api.services.supabase_query import parse_uuid


class PerformanceMetricsService:
    def __init__(self) -> None:
        self._history: list[dict[str, Any]] = []
        self._metrics = performance_metrics_repository

    @staticmethod
    def _item_key(item: dict[str, Any]) -> str:
        return str(item.get("id") or item.get("_id") or item.get("name") or "")

    def calculate_recall_at_k(self, ranked_items: list[dict[str, Any]], relevant_items: list[dict[str, Any]], k: int) -> float:
        if not ranked_items or not relevant_items:
            return 0.0
        top_k = ranked_items[:k]
        relevant_keys = {self._item_key(x) for x in relevant_items}
        hit = sum(1 for it in top_k if self._item_key(it) in relevant_keys)
        return round(hit / len(relevant_items), 2)

    def calculate_precision_at_k(
        self, ranked_items: list[dict[str, Any]], relevant_items: list[dict[str, Any]], k: int
    ) -> float:
        if not ranked_items or k <= 0:
            return 0.0
        top_k = ranked_items[:k]
        relevant_keys = {self._item_key(x) for x in relevant_items}
        hit = sum(1 for it in top_k if self._item_key(it) in relevant_keys)
        return round(hit / k, 2)

    def calculate_accuracy(self, generated_items: list[dict[str, Any]], approved_items: list[dict[str, Any]]) -> float:
        if not generated_items:
            return 0.0
        return round(len(approved_items) / len(generated_items), 2)

    def generate_performance_report(
        self,
        ranked_items: list[dict[str, Any]],
        approved_items: list[dict[str, Any]],
        rejected_items: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        rejected_items = rejected_items or []
        accuracy = self.calculate_accuracy(ranked_items, approved_items)
        return {
            "recallAt1": self.calculate_recall_at_k(ranked_items, approved_items, 1),
            "recallAt5": self.calculate_recall_at_k(ranked_items, approved_items, 5),
            "recallAt10": self.calculate_recall_at_k(ranked_items, approved_items, 10),
            "accuracy": accuracy,
            "precisionAt5": self.calculate_precision_at_k(ranked_items, approved_items, 5),
            "precisionAt10": self.calculate_precision_at_k(ranked_items, approved_items, 10),
            "totalGenerated": len(ranked_items),
            "totalApproved": len(approved_items),
            "totalRejected": len(rejected_items),
            "approvalRate": accuracy,
            "rejectionRate": round(len(rejected_items) / len(ranked_items), 2) if ranked_items else 0.0,
        }

    def track_performance_metrics(self, project_id: str, metric_type: str, metrics: dict[str, Any]) -> dict[str, Any]:
        project_id = parse_uuid(project_id, field="project id")
        payload = {"projectId": project_id, "type": metric_type, **metrics}
        self._history.append(payload)
        self._metrics.insert(project_id=project_id, metric_type=metric_type, payload=payload)
        return payload

    def get_performance_metrics_history(self, project_id: str, metric_type: str, limit: int = 10) -> list[dict[str, Any]]:
        project_id = parse_uuid(project_id, field="project id")
        result = self._metrics.list_recent(project_id=project_id, metric_type=metric_type, limit=limit)
        return [row.get("payload") or {} for row in (result.data or [])]


performance_metrics_service = PerformanceMetricsService()
