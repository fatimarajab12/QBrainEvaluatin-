"""Vector index adapter: FAISS + embeddings."""
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
