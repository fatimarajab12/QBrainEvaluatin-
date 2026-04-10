"""Vector index adapter: FAISS + embeddings.

By default the lab builds indexes **in memory** only. To clear optional on-disk caches, run
``python scripts/clear_vector_cache.py`` (removes ``data/faiss_cache/`` and LangChain ``index.faiss`` folders).
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from qbrain_rag.infrastructure.embeddings import get_embedding_model


def build_faiss_store(chunks: list[str], metadata_list: list[dict] | None = None):
    if metadata_list is None:
        metadata_list = [{} for _ in chunks]

    docs = [
        Document(page_content=chunk, metadata=meta)
        for chunk, meta in zip(chunks, metadata_list, strict=True)
    ]

    embeddings = get_embedding_model()
    return FAISS.from_documents(docs, embeddings)


def indexed_document_count(store: FAISS) -> int:
    """Number of vectors / documents in the FAISS index (one per chunk)."""
    return len(store.index_to_docstore_id)


def list_all_documents_ordered(store: FAISS) -> list[Document]:
    """
    Return every chunk in the index as a Document, ordered by ``chunk_id`` metadata when present
    (typical after ingestion), else by FAISS internal index order.
    """
    out: list[Document] = []
    for doc_id in store.index_to_docstore_id.values():
        doc = store.docstore.search(doc_id)
        if isinstance(doc, Document):
            out.append(doc)

    def _chunk_sort_key(d: Document) -> tuple[int, int | str]:
        cid = d.metadata.get("chunk_id")
        if isinstance(cid, int):
            return (0, cid)
        if isinstance(cid, str) and cid.isdigit():
            return (0, int(cid))
        try:
            return (0, int(cid))
        except (TypeError, ValueError):
            return (1, str(cid))

    out.sort(key=_chunk_sort_key)
    return out


def chunk_texts_materialized_in_store(store: FAISS, chunks: list[str]) -> bool:
    """
    Verify that every input chunk's text is stored in the FAISS docstore (not just vectors).

    LangChain keeps full `Document.page_content` in an in-memory docstore keyed by id; the FAISS
    index stores embedding vectors pointing at those ids.
    """
    if indexed_document_count(store) != len(chunks):
        return False
    texts_in_store: list[str] = []
    for doc_id in store.index_to_docstore_id.values():
        doc = store.docstore.search(doc_id)
        if not isinstance(doc, Document):
            return False
        texts_in_store.append(doc.page_content)
    return set(texts_in_store) == set(chunks) and len(texts_in_store) == len(chunks)


def save_faiss_store(store: FAISS, folder_path: str | Path, *, index_name: str = "index") -> None:
    """Persist FAISS vectors + docstore (including chunk text) to disk."""
    store.save_local(str(folder_path), index_name=index_name)


def load_faiss_store(folder_path: str | Path, *, index_name: str = "index") -> FAISS:
    """Load a store saved with `save_faiss_store` (needs the same embedding model as build time)."""
    embeddings = get_embedding_model()
    return FAISS.load_local(
        str(folder_path),
        embeddings,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )


def retrieve_top_k(store: FAISS, query: str, k: int | None = None) -> list[Document]:
    """Similarity search; ``k`` defaults to ``Settings.default_top_k``."""
    from qbrain_rag.config.settings import get_settings

    if k is None:
        k = get_settings().default_top_k
    return store.similarity_search(query, k=k)


def retrieve_top_k_for_source_files(
    store: FAISS,
    query: str,
    allowed_source_files: set[str] | list[str],
    k: int | None = None,
    *,
    fetch_multiplier: int = 12,
    max_fetch_cap: int | None = None,
) -> list[Document]:
    """
    Top-``k`` similarity search restricted to documents whose metadata ``source_file`` is in
    ``allowed_source_files``. FAISS has no native metadata filter, so we over-fetch globally
    then keep the first ``k`` hits that match (same idea as a single-SRS sub-index).

    **Widening:** A single small global window (e.g. ``k * 12``) often misses every chunk from
    the allowed file when many other PDFs share the index—the top vectors are all from other
    sources. This function **doubles** the fetch window until it collects ``k`` allowed hits or
    the whole index has been searched.

    If too few matches exist at all (or ``max_fetch_cap`` stops widening), returns as many as
    were found (may be fewer than ``k``).
    """
    from qbrain_rag.config.settings import get_settings

    if k is None:
        k = get_settings().default_top_k
    allowed = {str(x) for x in allowed_source_files}
    total = indexed_document_count(store)
    mult = max(2, int(fetch_multiplier))
    fetch_k = min(total, max(k * mult, k + 1))
    if max_fetch_cap is not None:
        fetch_k = min(fetch_k, int(max_fetch_cap))

    while True:
        ranked = store.similarity_search(query, k=fetch_k)
        out = []
        for doc in ranked:
            name = doc.metadata.get("source_file")
            if name is None:
                continue
            if str(name) in allowed:
                out.append(doc)
            if len(out) >= k:
                break
        if len(out) >= k or fetch_k >= total:
            return out[:k]
        if max_fetch_cap is not None and fetch_k >= int(max_fetch_cap):
            return out[:k]
        next_fetch = min(total, max(fetch_k * 2, fetch_k + 1))
        if max_fetch_cap is not None:
            next_fetch = min(next_fetch, int(max_fetch_cap))
        if next_fetch <= fetch_k:
            return out[:k]
        fetch_k = next_fetch
