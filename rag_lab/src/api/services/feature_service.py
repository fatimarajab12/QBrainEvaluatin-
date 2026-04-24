from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from api.controllers.rag_controller import RAGController
from api.repositories.feature_repository import feature_repository
from api.services.performance_metrics_service import performance_metrics_service
from api.services.supabase_query import parse_uuid
from api.services.vector_sync_service import delete_vectors_by_metadata, upsert_feature_vector


class FeatureService:
    def __init__(self) -> None:
        self._rag_controller = RAGController()
        self._features = feature_repository

    def persist_pipeline_results(
        self,
        project_id: str,
        pipeline_result: dict[str, Any],
        *,
        persist_test_cases: bool = True,
    ) -> dict[str, int]:
        """Persist features (and optionally test cases) produced by ``run_document_pipeline`` / RAG stack."""
        from api.services.test_case_service import test_case_service

        project_id = parse_uuid(project_id, field="project id")
        created_features = 0
        created_test_cases = 0
        for feature_item in pipeline_result.get("features", []) or []:
            feature = self.create_feature(
                {
                    "project_id": project_id,
                    "title": str(feature_item.get("name", feature_item.get("featureName", "feature"))),
                    "content": str(feature_item.get("description", "")),
                    "is_ai_generated": True,
                }
            )
            created_features += 1
            if persist_test_cases:
                for tc in feature_item.get("testCases", []) or []:
                    test_case_service.create_test_case(
                        {
                            "project_id": project_id,
                            "feature_id": feature["id"],
                            "title": str(tc.get("title", "test case")),
                            "steps": [str(s) for s in tc.get("steps", [])],
                            "expected_result": str(tc.get("expectedResult", "")),
                            "is_ai_generated": True,
                        }
                    )
                    created_test_cases += 1
        return {"feature_count": created_features, "test_case_count": created_test_cases}

    def create_feature(self, feature_data: dict[str, Any]) -> dict[str, Any]:
        project_id = parse_uuid(str(feature_data["project_id"]), field="project id")
        payload = {
            "project_id": project_id,
            "title": feature_data["title"],
            "content": feature_data.get("content"),
            "is_ai_generated": bool(feature_data.get("is_ai_generated", False)),
        }
        result = self._features.insert(payload)
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create feature")
        row = result.data[0]
        feature = {
            "id": str(row["id"]),
            "project_id": str(row["project_id"]),
            "title": row["title"],
            "content": row.get("content"),
            "is_ai_generated": bool(row.get("is_ai_generated", False)),
        }
        upsert_feature_vector(feature)
        return feature

    def get_feature_by_id(self, feature_id: str) -> dict[str, Any] | None:
        feature_id = parse_uuid(feature_id, field="feature id")
        result = self._features.fetch_by_id(feature_id)
        if not result.data:
            return None
        row = result.data[0]
        return {
            "id": str(row["id"]),
            "project_id": str(row["project_id"]),
            "title": row["title"],
            "content": row.get("content"),
            "is_ai_generated": bool(row.get("is_ai_generated", False)),
        }

    def has_ai_generated_features(self, project_id: str) -> bool:
        project_id = parse_uuid(project_id, field="project id")
        result = self._features.count_ai_by_project(project_id)
        return int(result.count or 0) > 0

    def get_project_features(self, project_id: str) -> list[dict[str, Any]]:
        project_id = parse_uuid(project_id, field="project id")
        result = self._features.list_by_project(project_id)
        return [
            {
                "id": str(row["id"]),
                "project_id": str(row["project_id"]),
                "title": row["title"],
                "content": row.get("content"),
                "is_ai_generated": bool(row.get("is_ai_generated", False)),
            }
            for row in (result.data or [])
        ]

    def update_feature(self, feature_id: str, update_data: dict[str, Any]) -> dict[str, Any]:
        feature_id = parse_uuid(feature_id, field="feature id")
        result = self._features.update(feature_id, update_data)
        if not result.data:
            raise HTTPException(status_code=404, detail="Feature not found")
        row = result.data[0]
        feature = {
            "id": str(row["id"]),
            "project_id": str(row["project_id"]),
            "title": row["title"],
            "content": row.get("content"),
            "is_ai_generated": bool(row.get("is_ai_generated", False)),
        }
        delete_vectors_by_metadata(feature["project_id"], "feature_id", feature["id"])
        upsert_feature_vector(feature)
        return feature

    def delete_feature(self, feature_id: str) -> dict[str, Any]:
        feature_id = parse_uuid(feature_id, field="feature id")
        feature = self.get_feature_by_id(feature_id)
        if not feature:
            raise HTTPException(status_code=404, detail="Feature not found")
        self._features.delete(feature_id)
        delete_vectors_by_metadata(feature["project_id"], "feature_id", feature_id)
        return {"success": True}

    def generate_features_from_srs(self, project_id: str, options: dict[str, Any]) -> dict[str, Any]:
        project_id = parse_uuid(project_id, field="project id")
        project = self._features.fetch_project_row(project_id)
        row = project.data[0] if project.data else None
        if not row and not options.get("doc_path"):
            raise HTTPException(status_code=404, detail="Project not found")
        doc_path = options.get("doc_path") or (row or {}).get("doc_path")
        if not doc_path:
            raise HTTPException(status_code=400, detail="No document path available for generation")
        skip_tests = bool(options.get("skip_tests", True))
        result = self._rag_controller.document_pipeline(
            doc_path=doc_path,
            test_context_k=int(options.get("test_context_k", 5)),
            max_features=None,
            skip_tests=skip_tests,
            quiet=True,
            project_id=project_id,
        )
        persisted = self.persist_pipeline_results(project_id, result, persist_test_cases=not skip_tests)
        return {**result, "persisted": persisted}

    def bulk_create_features(self, project_id: str, features_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parse_uuid(project_id, field="project id")
        return [self.create_feature({**x, "project_id": project_id}) for x in features_data]

    def get_test_cases_count(self, feature_id: str) -> dict[str, Any]:
        feature_id = parse_uuid(feature_id, field="feature id")
        result = self._features.count_test_cases_by_feature(feature_id)
        return {"featureId": feature_id, "testCasesCount": int(result.count or 0)}

    def get_performance_metrics(self, project_id: str) -> dict[str, Any]:
        parse_uuid(project_id, field="project id")
        items = self.get_project_features(project_id)
        ranked = [{"id": x["id"], "name": x["title"]} for x in items]
        approved = ranked[: min(5, len(ranked))]
        report = performance_metrics_service.generate_performance_report(ranked, approved)
        return performance_metrics_service.track_performance_metrics(project_id, "features", report)


feature_service = FeatureService()
