"""Vector index adapter: Supabase + embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import NAMESPACE_URL, UUID, uuid5

from langchain_core.documents import Document
import numpy as np

from api.services.supabase_client import get_supabase_client
from config.settings import get_settings
from infrastructure.embeddings import get_embedding_model


@dataclass
class SupabaseStore:
    project_id: str
    docs: list[Document]
    vectors: list[list[float]]

    def similarity_search_with_score(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """LangChain-style (doc, distance): lower distance means a better match."""
        embeddings = get_embedding_model()
        qvec = embeddings.embed_query(query)
        settings = get_settings()
        kk = max(1, int(k))
        if settings.use_supabase:
            client = get_supabase_client()
            rpc = client.rpc(
                "match_project_vectors",
                {
                    "in_project_id": self.project_id,
                    "query_embedding": qvec,
                    "match_count": kk,
                    "min_similarity": 0.0,
                },
            ).execute()
            rows = rpc.data or []
            out: list[tuple[Document, float]] = []
            for r in rows:
                sim = float(r.get("similarity", 0.0))
                dist = 1.0 - sim
                doc = Document(page_content=str(r.get("content", "")), metadata=r.get("metadata") or {})
                out.append((doc, dist))
            return out

        if not self.vectors:
            return []
        mat = np.array(self.vectors, dtype=float)
        q = np.array(qvec, dtype=float)
        denom = (np.linalg.norm(mat, axis=1) * np.linalg.norm(q)) + 1e-12
        sims = (mat @ q) / denom
        order = np.argsort(-sims)[:kk]
        return [(self.docs[int(i)], float(1.0 - sims[int(i)])) for i in order]


def _resolve_project_id(metadata_list: list[dict]) -> str:
    if not metadata_list:
        return str(uuid5(NAMESPACE_URL, "qbrain-global"))
    first = metadata_list[0]
    raw = str(first.get("project_id") or first.get("projectId") or first.get("source_file") or "qbrain-global")
    try:
        return str(UUID(raw))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, raw))


def build_vector_store(chunks: list[str], metadata_list: list[dict] | None = None) -> SupabaseStore:
    if metadata_list is None:
        metadata_list = [{} for _ in chunks]

    docs = [
        Document(page_content=chunk, metadata=meta)
        for chunk, meta in zip(chunks, metadata_list, strict=True)
    ]

    embeddings = get_embedding_model()
    vectors = embeddings.embed_documents([d.page_content for d in docs])
    project_id = _resolve_project_id(metadata_list)
    settings = get_settings()
    if settings.use_supabase:
        client = get_supabase_client()
        rows = [
            {
                "project_id": project_id,
                "content": doc.page_content,
                "embedding": vec,
                "metadata": doc.metadata,
            }
            for doc, vec in zip(docs, vectors, strict=True)
        ]
        if rows:
            client.table("project_vectors").insert(rows).execute()
    return SupabaseStore(project_id=project_id, docs=docs, vectors=vectors)


def indexed_document_count(store: SupabaseStore) -> int:
    return len(store.docs)


def list_all_documents_ordered(store: SupabaseStore) -> list[Document]:
    out = list(store.docs)
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


def chunk_texts_materialized_in_store(store: SupabaseStore, chunks: list[str]) -> bool:
    if indexed_document_count(store) != len(chunks):
        return False
    texts_in_store = [d.page_content for d in store.docs]
    return set(texts_in_store) == set(chunks) and len(texts_in_store) == len(chunks)


def retrieve_top_k(store: SupabaseStore, query: str, k: int | None = None) -> list[Document]:
    if k is None:
        k = get_settings().default_top_k
    embeddings = get_embedding_model()
    qvec = embeddings.embed_query(query)
    settings = get_settings()
    if settings.use_supabase:
        client = get_supabase_client()
        rpc = client.rpc(
            "match_project_vectors",
            {
                "in_project_id": store.project_id,
                "query_embedding": qvec,
                "match_count": int(k),
                "min_similarity": 0.0,
            },
        ).execute()
        rows = rpc.data or []
        if rows:
            return [
                Document(page_content=str(r.get("content", "")), metadata=r.get("metadata") or {})
                for r in rows
            ]

    if not store.vectors:
        return []
    mat = np.array(store.vectors, dtype=float)
    q = np.array(qvec, dtype=float)
    denom = (np.linalg.norm(mat, axis=1) * np.linalg.norm(q)) + 1e-12
    sims = (mat @ q) / denom
    order = np.argsort(-sims)[: int(k)]
    return [store.docs[int(i)] for i in order]


def retrieve_top_k_for_source_files(
    store: SupabaseStore,
    query: str,
    allowed_source_files: set[str] | list[str],
    k: int | None = None,
    *,
    fetch_multiplier: int = 12,
    max_fetch_cap: int | None = None,
) -> list[Document]:
    if k is None:
        k = get_settings().default_top_k
    allowed = {str(x) for x in allowed_source_files}
    total = indexed_document_count(store)
    mult = max(2, int(fetch_multiplier))
    fetch_k = min(total, max(k * mult, k + 1))
    if max_fetch_cap is not None:
        fetch_k = min(fetch_k, int(max_fetch_cap))

    while True:
        ranked = retrieve_top_k(store, query, k=fetch_k)
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
