from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from api.controllers.rag_controller import RAGController
from api.repositories.test_case_repository import test_case_repository
from api.services.supabase_query import parse_uuid
from api.services.vector_sync_service import delete_vectors_by_metadata, upsert_test_case_vector


def _to_gherkin(test_case: dict[str, Any]) -> str:
    steps = "\n".join(f"    And {s}" for s in test_case.get("steps", []))
    return (
        f"Feature: {test_case.get('title', 'Test Feature')}\n"
        f"  Scenario: {test_case.get('title', 'Test Scenario')}\n"
        f"{steps}\n"
        f"    Then {test_case.get('expected_result', '')}"
    )


class TestCaseService:
    def __init__(self) -> None:
        self._rag_controller = RAGController()
        self._test_cases = test_case_repository

    def create_test_case(self, test_case_data: dict[str, Any]) -> dict[str, Any]:
        project_id = parse_uuid(str(test_case_data["project_id"]), field="project id")
        feature_id = parse_uuid(str(test_case_data["feature_id"]), field="feature id")
        payload = {
            "project_id": project_id,
            "feature_id": feature_id,
            "title": test_case_data["title"],
            "steps": test_case_data.get("steps", []),
            "expected_result": test_case_data.get("expected_result", ""),
            "is_ai_generated": bool(test_case_data.get("is_ai_generated", False)),
        }
        result = self._test_cases.insert(payload)
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create test case")
        row = result.data[0]
        test_case = {
            "id": str(row["id"]),
            "project_id": str(row["project_id"]),
            "feature_id": str(row["feature_id"]),
            "title": row["title"],
            "steps": row.get("steps") or [],
            "expected_result": row.get("expected_result") or "",
            "is_ai_generated": bool(row.get("is_ai_generated", False)),
        }
        upsert_test_case_vector(test_case)
        return test_case

    def get_test_case_by_id(self, test_case_id: str) -> dict[str, Any] | None:
        test_case_id = parse_uuid(test_case_id, field="test case id")
        result = self._test_cases.fetch_by_id(test_case_id)
        if not result.data:
            return None
        row = result.data[0]
        return {
            "id": str(row["id"]),
            "project_id": str(row["project_id"]),
            "feature_id": str(row["feature_id"]),
            "title": row["title"],
            "steps": row.get("steps") or [],
            "expected_result": row.get("expected_result") or "",
            "is_ai_generated": bool(row.get("is_ai_generated", False)),
        }

    def has_ai_generated_test_cases(self, feature_id: str) -> bool:
        feature_id = parse_uuid(feature_id, field="feature id")
        result = self._test_cases.count_ai_by_feature(feature_id)
        return int(result.count or 0) > 0

    def get_feature_test_cases(self, feature_id: str) -> list[dict[str, Any]]:
        feature_id = parse_uuid(feature_id, field="feature id")
        result = self._test_cases.list_by_feature(feature_id)
        return [
            {
                "id": str(row["id"]),
                "project_id": str(row["project_id"]),
                "feature_id": str(row["feature_id"]),
                "title": row["title"],
                "steps": row.get("steps") or [],
                "expected_result": row.get("expected_result") or "",
                "is_ai_generated": bool(row.get("is_ai_generated", False)),
            }
            for row in (result.data or [])
        ]

    def get_project_test_cases(self, project_id: str) -> list[dict[str, Any]]:
        project_id = parse_uuid(project_id, field="project id")
        result = self._test_cases.list_by_project(project_id)
        return [
            {
                "id": str(row["id"]),
                "project_id": str(row["project_id"]),
                "feature_id": str(row["feature_id"]),
                "title": row["title"],
                "steps": row.get("steps") or [],
                "expected_result": row.get("expected_result") or "",
                "is_ai_generated": bool(row.get("is_ai_generated", False)),
            }
            for row in (result.data or [])
        ]

    def get_all_test_cases_by_features(self, project_id: str) -> dict[str, Any]:
        project_id = parse_uuid(project_id, field="project id")
        features_result = self._test_cases.list_features_by_project(project_id)
        features = [
            {
                "id": str(row["id"]),
                "project_id": str(row["project_id"]),
                "title": row["title"],
                "content": row.get("content"),
                "is_ai_generated": bool(row.get("is_ai_generated", False)),
            }
            for row in (features_result.data or [])
        ]
        return {
            "projectId": project_id,
            "features": [
                {"feature": feature, "testCases": self.get_feature_test_cases(feature["id"])}
                for feature in features
            ],
        }

    def update_test_case(self, test_case_id: str, update_data: dict[str, Any]) -> dict[str, Any]:
        test_case_id = parse_uuid(test_case_id, field="test case id")
        result = self._test_cases.update(test_case_id, update_data)
        if not result.data:
            raise HTTPException(status_code=404, detail="Test case not found")
        row = result.data[0]
        test_case = {
            "id": str(row["id"]),
            "project_id": str(row["project_id"]),
            "feature_id": str(row["feature_id"]),
            "title": row["title"],
            "steps": row.get("steps") or [],
            "expected_result": row.get("expected_result") or "",
            "is_ai_generated": bool(row.get("is_ai_generated", False)),
        }
        delete_vectors_by_metadata(test_case["project_id"], "test_case_id", test_case["id"])
        upsert_test_case_vector(test_case)
        return test_case

    def delete_test_case(self, test_case_id: str) -> dict[str, Any]:
        test_case_id = parse_uuid(test_case_id, field="test case id")
        test_case = self.get_test_case_by_id(test_case_id)
        if not test_case:
            raise HTTPException(status_code=404, detail="Test case not found")
        self._test_cases.delete(test_case_id)
        delete_vectors_by_metadata(test_case["project_id"], "test_case_id", test_case_id)
        return {"success": True}

    def generate_test_cases_for_feature(self, feature_id: str, options: dict[str, Any]) -> list[dict[str, Any]]:
        feature_id = parse_uuid(feature_id, field="feature id")
        feature = self._test_cases.fetch_feature_row(feature_id)
        row = feature.data[0] if feature.data else None
        feature = (
            {
                "id": str(row["id"]),
                "project_id": str(row["project_id"]),
                "title": row["title"],
                "content": row.get("content"),
            }
            if row
            else None
        )
        if not feature:
            raise HTTPException(status_code=404, detail="Feature not found")
        project = self._test_cases.fetch_project_row(feature["project_id"])
        project_row = project.data[0] if project.data else None
        doc_path = options.get("doc_path") or (project_row or {}).get("doc_path")
        if not doc_path:
            raise HTTPException(status_code=400, detail="No document path available for generation")
        result = self._rag_controller.document_pipeline(
            doc_path=doc_path,
            test_context_k=int(options.get("test_context_k", 5)),
            max_features=None,
            skip_tests=False,
            quiet=True,
            project_id=feature["project_id"],
        )
        created: list[dict[str, Any]] = []
        for item in result.get("features", []):
            for tc in item.get("testCases", []):
                created.append(
                    self.create_test_case(
                        {
                            "project_id": feature["project_id"],
                            "feature_id": feature_id,
                            "title": str(tc.get("title", "test case")),
                            "steps": [str(s) for s in tc.get("steps", [])],
                            "expected_result": str(tc.get("expectedResult", "")),
                            "is_ai_generated": True,
                        }
                    )
                )
        return created

    def bulk_create_test_cases(self, feature_id: str, test_cases_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        feature_id = parse_uuid(feature_id, field="feature id")
        feature = self._test_cases.fetch_feature_row(feature_id)
        row = feature.data[0] if feature.data else None
        feature = {"project_id": str(row["project_id"])} if row else None
        if not feature:
            raise HTTPException(status_code=404, detail="Feature not found")
        return [
            self.create_test_case(
                {
                    **item,
                    "project_id": feature["project_id"],
                    "feature_id": feature_id,
                }
            )
            for item in test_cases_data
        ]

    def convert_test_case_to_gherkin(self, test_case_id: str) -> dict[str, Any]:
        test_case_id = parse_uuid(test_case_id, field="test case id")
        test_case = self.get_test_case_by_id(test_case_id)
        if not test_case:
            raise HTTPException(status_code=404, detail="Test case not found")
        return {"id": test_case_id, "gherkin": _to_gherkin(test_case)}


test_case_service = TestCaseService()
