"""Turn lists of LangChain ``Document`` into one string for the LLM (compact chunk markers)."""
from __future__ import annotations

from langchain_core.documents import Document

_SEP = "\n\n---\n\n"
_TRUNC_SUFFIX = "\n\n[... context truncated ...]"


def documents_to_context_string(
    documents: list[Document],
    *,
    max_chars: int | None = None,
) -> tuple[str, bool, int]:
    """
    Join chunk texts with ``#chunk_id`` headers. If several ``source_file`` values exist, adds
    ``@filename`` to disambiguate.

    When ``max_chars`` is set, truncation happens at **chunk boundaries** (no chunk is cut in half),
    except when a single chunk alone exceeds the budget — then that chunk is character-truncated.

    Returns ``(text, truncated, n_chunks_fully_included)``.
    """
    sources = {d.metadata.get("source_file") for d in documents if d.metadata.get("source_file")}
    multi_source = len(sources) > 1

    def _header(i: int, d: Document) -> str:
        cid = d.metadata.get("chunk_id", i + 1)
        src = d.metadata.get("source_file", "")
        if multi_source and src:
            return f"#{cid} @{src}\n"
        return f"#{cid}\n"

    blocks = [_header(i, d) + d.page_content for i, d in enumerate(documents)]

    if max_chars is None:
        return _SEP.join(blocks), False, len(blocks)

    suffix_len = len(_TRUNC_SUFFIX)
    budget = max_chars - suffix_len
    if budget <= 0:
        return _TRUNC_SUFFIX[: max_chars], True, 0

    out: list[str] = []
    used = 0
    n_full = 0

    for block in blocks:
        sep_len = len(_SEP) if out else 0
        need = sep_len + len(block)
        if used + need <= budget:
            if out:
                used += sep_len
            used += len(block)
            out.append(block)
            n_full += 1
            continue
        if not out:
            partial = block[:budget]
            return partial + _TRUNC_SUFFIX, True, 0
        return _SEP.join(out) + _TRUNC_SUFFIX, True, n_full

    return _SEP.join(out), False, n_full
