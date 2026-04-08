"""Feature extraction: all indexed chunks → one LLM context (bounded by ``max_context_chars``)."""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from qbrain_rag.application.document_context import documents_to_context_string
from qbrain_rag.application.prompts_srs import (
    FEATURE_EXTRACTION_USER_TEMPLATE,
    create_adaptive_prompt,
)
from qbrain_rag.infrastructure.llm import complete_json_object
from qbrain_rag.infrastructure.vector_store import list_all_documents_ordered

_FEATURE_SYSTEM = (
    "You are an expert software requirements analyst. "
    "Extract testable features grounded in the user’s DOCUMENT CONTEXT only; follow the JSON shape and field names in the user message exactly. "
    "The context contains all chunks of the indexed file in reading order (or a truncated prefix if stated). "
    "Cover obligations from the entire context, not only the opening. Do not require a formal document template. "
    "Apply the consolidation rules in the user message."
)

_DEFAULT_MAX_CONTEXT_CHARS = 120_000


def extract_features_from_indexed_chunks(
    store,
    *,
    max_context_chars: int | None = _DEFAULT_MAX_CONTEXT_CHARS,
) -> dict[str, Any]:
    """
    Concatenate **all** chunks from the store (reading order), optionally capped at chunk boundaries.

    This is **not** similarity-based retrieval; the index is used so later stages can run ``similarity_search``.
    """
    documents: list[Document] = list_all_documents_ordered(store)
    if not documents:
        raise ValueError("No document content in index. Ingest a file first.")

    context, truncated, n_full = documents_to_context_string(
        documents,
        max_chars=max_context_chars,
    )
    adaptive = create_adaptive_prompt()
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
    parsed = complete_json_object(_FEATURE_SYSTEM, user, temperature=0.3)
    raw_features = parsed.get("features")
    if not isinstance(raw_features, list):
        raw_features = []

    features: list[dict[str, Any]] = [item for item in raw_features if isinstance(item, dict)]

    return {
        "features": features,
        "metadata": {
            "total_chunks": len(documents),
            "chunks_fully_in_context": n_full,
            "context_truncated": truncated,
        },
    }
