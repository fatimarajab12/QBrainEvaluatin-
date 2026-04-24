"""Feature extraction: indexed chunks → LLM (single pass for short docs; segment + merge for long)."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.documents import Document

from application.document_context import documents_to_context_string
from application.prompts_srs import (
    FEATURE_CONSOLIDATION_USER_TEMPLATE,
    FEATURE_EXTRACTION_USER_TEMPLATE,
    FEATURE_PARTIAL_USER_TEMPLATE,
    create_adaptive_prompt,
)
from infrastructure.llm import complete_json_object
from infrastructure.vector_store import list_all_documents_ordered

_FEATURE_SYSTEM = (
    "You are an expert software requirements analyst. "
    "Extract testable features grounded in the user’s DOCUMENT CONTEXT only; follow the JSON shape and field names in the user message exactly. "
    "The context contains all chunks of the indexed file in reading order (or a truncated prefix if stated). "
    "Cover obligations from the entire context, not only the opening. Do not require a formal document template. "
    "Apply the consolidation rules in the user message."
)

_PARTIAL_SYSTEM = (
    "You are an expert software requirements analyst. "
    "The user’s CONTEXT is **one segment** of a longer indexed document (see segment statistics). "
    "Extract testable features grounded **only** in this segment; follow the JSON shape in the user message exactly. "
    "Do not invent obligations from unseen parts of the file."
)

_CONSOLIDATION_SYSTEM = (
    "You merge partial feature lists from sequential document segments into one deduplicated list. "
    "Follow the JSON shape and field names in the user message exactly. Output one JSON object only, no markdown. "
    "Do not add requirements not supported by the candidate features."
)

_DEFAULT_MAX_CONTEXT_CHARS = 120_000
_DEFAULT_CHUNKS_PER_GROUP = 5
_EXTRACTION_TEMPERATURE = 0.1


def _chunk_marker_id(documents: list[Document], index: int) -> int | str:
    cid = documents[index].metadata.get("chunk_id")
    if cid is not None:
        return cid
    return index + 1


def _parse_features_list(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    raw = parsed.get("features")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def extract_features_from_indexed_chunks(
    store,
    *,
    max_context_chars: int | None = _DEFAULT_MAX_CONTEXT_CHARS,
    chunks_per_group: int = _DEFAULT_CHUNKS_PER_GROUP,
    temperature: float = _EXTRACTION_TEMPERATURE,
) -> dict[str, Any]:
    """
    Load **all** chunks in reading order. For small documents (≤ ``chunks_per_group`` chunks), one LLM call
    with the full concatenated context (optionally capped). For longer documents, extract per segment then
    merge/deduplicate in a second pass.
    """
    documents: list[Document] = list_all_documents_ordered(store)
    if not documents:
        raise ValueError("No document content in index. Ingest a file first.")

    k = max(1, int(chunks_per_group))
    adaptive = create_adaptive_prompt()

    if len(documents) <= k:
        return _extract_single_pass(
            documents,
            adaptive=adaptive,
            max_context_chars=max_context_chars,
            temperature=temperature,
        )

    return _extract_segment_then_merge(
        documents,
        adaptive=adaptive,
        chunks_per_group=k,
        max_context_chars=max_context_chars,
        temperature=temperature,
    )


def _extract_single_pass(
    documents: list[Document],
    *,
    adaptive: str,
    max_context_chars: int | None,
    temperature: float,
) -> dict[str, Any]:
    context, truncated, n_full = documents_to_context_string(
        documents,
        max_chars=max_context_chars,
    )
    if truncated and n_full == 0:
        trunc_detail = "truncated=yes (only a prefix of the first chunk fits the limit)."
    elif truncated:
        trunc_detail = f"truncated=yes ({n_full} full chunk(s) included; remaining chunks omitted)."
    else:
        trunc_detail = "truncated=no."
    context_stats = (
        f"**Context statistics:** {len(documents)} chunks in index; {trunc_detail}"
    )
    user = FEATURE_EXTRACTION_USER_TEMPLATE.format(
        adaptive_prompt=adaptive,
        context_stats=context_stats,
        context=context,
    )
    parsed = complete_json_object(_FEATURE_SYSTEM, user, temperature=temperature)
    features = _parse_features_list(parsed)
    return {
        "features": features,
        "llm_outputs": {
            "extraction_calls": [
                {
                    "stage": "single_pass_extraction",
                    "segment_index": 0,
                    "chunk_id_range": [
                        _chunk_marker_id(documents, 0),
                        _chunk_marker_id(documents, len(documents) - 1),
                    ],
                    "raw_json_object": parsed,
                }
            ],
            "consolidation_output": None,
        },
        "metadata": {
            "total_chunks": len(documents),
            "chunks_fully_in_context": n_full,
            "context_truncated": truncated,
            "extraction_mode": "single_pass",
            "chunks_per_group": None,
            "segment_count": 1,
        },
    }


def _extract_segment_then_merge(
    documents: list[Document],
    *,
    adaptive: str,
    chunks_per_group: int,
    max_context_chars: int | None,
    temperature: float,
) -> dict[str, Any]:
    batches: list[dict[str, Any]] = []
    llm_extraction_calls: list[dict[str, Any]] = []
    any_truncated = False

    for seg_idx, start in enumerate(range(0, len(documents), chunks_per_group)):
        group = documents[start : start + chunks_per_group]
        context, truncated, n_full = documents_to_context_string(
            group,
            max_chars=max_context_chars,
        )
        if truncated:
            any_truncated = True
        first_id = _chunk_marker_id(group, 0)
        last_id = _chunk_marker_id(group, len(group) - 1)
        if truncated and n_full == 0:
            trunc_detail = "truncated=yes (prefix of first chunk in this segment only)."
        elif truncated:
            trunc_detail = f"truncated=yes ({n_full} full chunk(s) in segment; rest of segment omitted)."
        else:
            trunc_detail = "truncated=no."
        context_stats = (
            f"segment {seg_idx + 1}; chunks in this segment: {len(group)}; "
            f"chunk markers from #{first_id} through #{last_id}; {trunc_detail}"
        )
        user = FEATURE_PARTIAL_USER_TEMPLATE.format(
            adaptive_prompt=adaptive,
            context_stats=context_stats,
            context=context,
        )
        parsed = complete_json_object(_PARTIAL_SYSTEM, user, temperature=temperature)
        feats = _parse_features_list(parsed)
        llm_extraction_calls.append(
            {
                "stage": "segment_extraction",
                "segment_index": seg_idx,
                "chunk_id_range": [first_id, last_id],
                "raw_json_object": parsed,
            }
        )
        batches.append(
            {
                "segment_index": seg_idx,
                "chunk_id_range": [first_id, last_id],
                "features": feats,
            }
        )

    candidates_json = json.dumps(batches, ensure_ascii=False)
    cons_user = FEATURE_CONSOLIDATION_USER_TEMPLATE.format(
        candidates_json=candidates_json,
    )
    merged = complete_json_object(_CONSOLIDATION_SYSTEM, cons_user, temperature=temperature)
    features = _parse_features_list(merged)

    return {
        "features": features,
        "llm_outputs": {
            "extraction_calls": llm_extraction_calls,
            "consolidation_output": {
                "stage": "consolidation",
                "raw_json_object": merged,
            },
        },
        "metadata": {
            "total_chunks": len(documents),
            "chunks_fully_in_context": len(documents),
            "context_truncated": any_truncated,
            "extraction_mode": "segment_then_merge",
            "chunks_per_group": chunks_per_group,
            "segment_count": len(batches),
        },
    }
