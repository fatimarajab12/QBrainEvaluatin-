"""Per-feature test cases using merged multi-query similarity retrieval over the index."""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from qbrain_rag.application.document_context import documents_to_context_string
from qbrain_rag.application.prompts_srs import build_test_case_user_prompt
from qbrain_rag.infrastructure.llm import complete_json_object

_TC_SYSTEM = (
    "You are an expert QA engineer. Generate concrete, test cases grounded in the user’s SOURCE CONTEXT only. "
    "Obey the JSON shape in the user message exactly. Output one JSON object only, no markdown."
)

_DEFAULT_MAX_TEST_CONTEXT = 12_000
_DEFAULT_N_CONTEXT_CHUNKS = 5
_K_PER_QUERY_MULT = 2

# Lexical rerank on top of FAISS L2 (lower = better): nudge distance down when chunk text hits name/anchors.
_NAME_IN_CHUNK_BONUS = 0.05
_MATCHED_PHRASE_IN_CHUNK_BONUS = 0.03
_MIN_MATCHED_LEN = 3
_MAX_MATCHED_FOR_BOOST = 20

# Few-shot example title pattern — reject if model copies it verbatim-ish
_GENERIC_TITLE_SUBSTRING = "verify required behavior under valid conditions"


def _doc_merge_key(doc: Document) -> tuple[Any, ...]:
    return (doc.metadata.get("chunk_id"), doc.metadata.get("source_file"))


def _boosted_distance(
    doc: Document,
    base_score: float,
    *,
    feature_name: str,
    matched_strs: list[str],
) -> float:
    """Lower is better (FAISS L2). Subtract small bonuses when chunk text overlaps name/anchors."""
    adj = float(base_score)
    body = doc.page_content.lower()
    nl = feature_name.strip().lower()
    if nl and nl in body:
        adj -= _NAME_IN_CHUNK_BONUS
    for m in matched_strs[:_MAX_MATCHED_FOR_BOOST]:
        ml = str(m).strip().lower()
        if len(ml) >= _MIN_MATCHED_LEN and ml in body:
            adj -= _MATCHED_PHRASE_IN_CHUNK_BONUS
    return adj


def _retrieve_merged_unique(
    store,
    labeled_queries: list[tuple[str, str]],
    *,
    k_per_query: int,
    max_unique: int,
    feature_name: str = "",
    matched_strs: list[str] | None = None,
) -> tuple[list[Document], dict[str, str], list[int | str]]:
    """
    Run similarity search per query; keep each chunk once, using best (lowest) FAISS L2 distance,
    then rerank unique chunks with a small lexical boost (name + matched sections in chunk text).
    """
    matched_strs = matched_strs or []
    best: dict[tuple[Any, ...], tuple[Document, float]] = {}
    queries_out: dict[str, str] = {}

    for label, q in labeled_queries:
        text = q.strip()
        if not text:
            continue
        queries_out[label] = text
        pairs = store.similarity_search_with_score(text, k=k_per_query)
        for doc, score in pairs:
            key = _doc_merge_key(doc)
            prev = best.get(key)
            if prev is None or score < prev[1]:
                best[key] = (doc, float(score))

    ranked = sorted(
        best.values(),
        key=lambda item: _boosted_distance(
            item[0],
            item[1],
            feature_name=feature_name,
            matched_strs=matched_strs,
        ),
    )
    chosen = [d for d, _ in ranked[:max_unique]]
    ids: list[int | str] = []
    for d in chosen:
        cid = d.metadata.get("chunk_id")
        ids.append(cid if cid is not None else "")
    return chosen, queries_out, ids


def _build_retrieval_queries(
    *,
    name: str,
    desc: str,
    matched: list[str],
) -> list[tuple[str, str]]:
    matched_join = " ".join(str(s) for s in matched[:12]).strip()
    out: list[tuple[str, str]] = []

    if name:
        out.append(("by_name", name))
    if name and matched_join:
        q = f"{name} {matched_join}"
        if q.strip() != name.strip():
            out.append(("by_name_and_matched_sections", q))
    elif matched_join and not name:
        out.append(("by_matched_sections", matched_join))
    if desc:
        dtrim = desc[:3000].strip()
        if dtrim and dtrim != name.strip():
            out.append(("by_description", dtrim))

    # de-duplicate identical query strings while preserving first label
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for label, q in out:
        if q in seen:
            continue
        seen.add(q)
        deduped.append((label, q))
    return deduped


def _title_too_generic(title: str) -> bool:
    t = title.strip().lower()
    if len(t) < 12:
        return True
    if _GENERIC_TITLE_SUBSTRING in t:
        return True
    if t in ("test", "happy path", "positive test", "negative test", "smoke test"):
        return True
    return False


def _test_case_passes_quality(tc: dict[str, Any]) -> bool:
    steps = tc.get("steps")
    if not isinstance(steps, list) or len(steps) < 3:
        return False
    exp = str(tc.get("expectedResult") or "").strip()
    if not exp:
        return False
    if _title_too_generic(str(tc.get("title") or "")):
        return False
    return True


def generate_test_cases_for_feature(
    store,
    feature: dict[str, Any],
    *,
    n_context_chunks: int = _DEFAULT_N_CONTEXT_CHUNKS,
    max_context_chars: int = _DEFAULT_MAX_TEST_CONTEXT,
) -> dict[str, Any]:
    """
    Returns ``{"testCases": [...], "evidence": {...}}`` with retrieval traceability for the feature
    (and duplicated minimal evidence on each test case for exports).
    """
    name = str(feature.get("name") or "").strip()
    desc = str(feature.get("description") or "").strip()
    feature_description = f"{name}\n{desc}".strip() or name or desc
    feature_type = str(feature.get("featureType") or "FUNCTIONAL")
    matched = feature.get("matchedSections") or []
    if not isinstance(matched, list):
        matched = []
    matched_strs = [str(x) for x in matched]

    labeled = _build_retrieval_queries(name=name, desc=desc, matched=matched_strs)
    if not labeled:
        labeled = [("fallback", feature_description[:3000] or "feature")]

    n = max(int(n_context_chunks), 4)
    k_each = max(n * _K_PER_QUERY_MULT, n)
    docs, queries_used, chunk_ids = _retrieve_merged_unique(
        store,
        labeled,
        k_per_query=k_each,
        max_unique=n,
        feature_name=name,
        matched_strs=matched_strs,
    )

    context, _, _ = documents_to_context_string(docs, max_chars=max_context_chars)

    user = build_test_case_user_prompt(
        feature_description=feature_description,
        context=context,
        feature_type=feature_type,
        matched_sections=matched_strs,
    )
    parsed = complete_json_object(_TC_SYSTEM, user, temperature=0.1)
    raw = parsed.get("testCases")
    if not isinstance(raw, list):
        raw = []

    per_tc_evidence = {
        "retrieved_chunk_ids": chunk_ids,
        "matched_sections": list(matched_strs),
    }
    feature_evidence = {
        **per_tc_evidence,
        "queries": queries_used,
    }

    normalized: list[dict[str, Any]] = []
    for i, tc in enumerate(raw):
        if not isinstance(tc, dict):
            continue
        title = str(tc.get("title") or "").strip()
        if not title:
            continue
        candidate = {
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
            "evidence": dict(per_tc_evidence),
        }
        if _test_case_passes_quality(candidate):
            normalized.append(candidate)

    seen_titles: set[str] = set()
    diverse: list[dict[str, Any]] = []
    for tc in normalized:
        tkey = str(tc.get("title") or "").strip().lower()
        if not tkey or tkey in seen_titles:
            continue
        seen_titles.add(tkey)
        diverse.append(tc)
    for j, tc in enumerate(diverse):
        tc["testCaseId"] = f"TC_{j + 1:03d}"
    normalized = diverse

    return {
        "testCases": normalized,
        "evidence": feature_evidence,
    }
