from __future__ import annotations

from typing import Any

from api.repositories.vector_repository import vector_repository
from api.services.supabase_query import parse_uuid
from infrastructure.embeddings import get_embedding_model


def _upsert_vector(project_id: str, content: str, metadata: dict[str, Any]) -> None:
    project_id = parse_uuid(project_id, field="project id")
    embedding = get_embedding_model().embed_query(content)
    row = {
        "project_id": project_id,
        "content": content,
        "embedding": embedding,
        "metadata": metadata,
    }
    vector_repository.insert_chunk(row)


def upsert_feature_vector(feature: dict[str, Any]) -> None:
    project_id = str(feature["project_id"])
    content = (
        f"Feature: {feature.get('title', '')}\n\n"
        f"Description: {feature.get('content', '')}"
    )
    _upsert_vector(
        project_id,
        content,
        {
            "type": "feature",
            "feature_id": str(feature["id"]),
            "is_ai_generated": bool(feature.get("is_ai_generated", False)),
        },
    )


def upsert_test_case_vector(test_case: dict[str, Any]) -> None:
    project_id = str(test_case["project_id"])
    steps = test_case.get("steps") or []
    steps_text = "\n".join(f"{idx + 1}. {s}" for idx, s in enumerate(steps))
    content = (
        f"Test Case: {test_case.get('title', '')}\n\n"
        f"Steps:\n{steps_text}\n\n"
        f"Expected Result: {test_case.get('expected_result', '')}"
    )
    _upsert_vector(
        project_id,
        content,
        {
            "type": "testcase",
            "test_case_id": str(test_case["id"]),
            "feature_id": str(test_case["feature_id"]),
            "is_ai_generated": bool(test_case.get("is_ai_generated", False)),
        },
    )


def delete_vectors_by_metadata(project_id: str, key: str, value: str) -> None:
    project_id = parse_uuid(project_id, field="project id")
    vector_repository.delete_by_metadata(project_id=project_id, metadata_filter={key: value})
