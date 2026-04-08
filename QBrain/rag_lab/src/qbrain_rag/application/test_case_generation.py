"""Per-feature test cases using top-k similarity retrieval over the index."""
from __future__ import annotations

from typing import Any

from qbrain_rag.application.document_context import documents_to_context_string
from qbrain_rag.application.prompts_srs import build_test_case_user_prompt
from qbrain_rag.infrastructure.llm import complete_json_object

_TC_SYSTEM = (
    "You are an expert QA engineer. Generate concrete, testable test cases grounded in the user’s SOURCE CONTEXT only. "
    "The text may be unstructured; do not assume sections or IDs unless they appear in the context. "
    "Obey the JSON shape in the user message exactly. Output one JSON object only, no markdown."
)

_DEFAULT_MAX_TEST_CONTEXT = 24_000


def generate_test_cases_for_feature(
    store,
    feature: dict[str, Any],
    *,
    n_context_chunks: int = 8,
    max_context_chars: int = _DEFAULT_MAX_TEST_CONTEXT,
) -> list[dict[str, Any]]:
    name = str(feature.get("name") or "").strip()
    desc = str(feature.get("description") or "").strip()
    feature_description = f"{name}\n{desc}".strip() or name or desc
    feature_type = str(feature.get("featureType") or "FUNCTIONAL")
    matched = feature.get("matchedSections") or []
    if not isinstance(matched, list):
        matched = []

    parts = [feature_description[:3000]]
    if matched:
        parts.insert(0, " ".join(str(s) for s in matched[:12]))
    query = " ".join(parts).strip()
    k = max(n_context_chunks, 4)
    docs = store.similarity_search(query, k=k)
    context, _, _ = documents_to_context_string(docs, max_chars=max_context_chars)

    user = build_test_case_user_prompt(
        feature_description=feature_description,
        context=context,
        feature_type=feature_type,
        matched_sections=[str(x) for x in matched],
    )
    parsed = complete_json_object(_TC_SYSTEM, user, temperature=0.3)
    raw = parsed.get("testCases")
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, Any]] = []
    for i, tc in enumerate(raw):
        if not isinstance(tc, dict):
            continue
        title = str(tc.get("title") or "").strip()
        if not title:
            continue
        normalized.append(
            {
                "testCaseId": str(tc.get("testCaseId") or f"TC_{i + 1:03d}"),
                "title": title,
                "description": str(tc.get("description") or ""),
                "steps": [str(s) for s in tc.get("steps", [])] if isinstance(tc.get("steps"), list) else [],
                "expectedResult": str(tc.get("expectedResult") or ""),
                "priority": str(tc.get("priority") or "medium").lower(),
                "status": str(tc.get("status") or "pending"),
                "preconditions": [str(p) for p in tc.get("preconditions", [])]
                if isinstance(tc.get("preconditions"), list)
                else [],
                "testData": tc.get("testData") if isinstance(tc.get("testData"), dict) else {},
            }
        )
    return normalized
