"""Evaluation metrics (embeddings-based similarity + optional IR helpers)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from qbrain_rag.infrastructure.embeddings import get_embedding_model

# Default binary “correct” cutoff for QA tables / notebooks (override via CLI `--threshold`).
SEMANTIC_SIMILARITY_DEFAULT_THRESHOLD = 0.70


def semantic_similarity(text1: str, text2: str) -> float:
    emb = get_embedding_model()
    v1 = np.array(emb.embed_query(text1)).reshape(1, -1)
    v2 = np.array(emb.embed_query(text2)).reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])


def passes_semantic_threshold(
    similarity: float,
    *,
    threshold: float | None = None,
) -> bool:
    """True if embedding cosine similarity meets the threshold (e.g. ``gen_correct`` columns)."""
    t = SEMANTIC_SIMILARITY_DEFAULT_THRESHOLD if threshold is None else threshold
    return float(similarity) >= t


def precision_at_k(retrieved_sections: list, relevant_sections: list) -> float:
    retrieved = {x for x in retrieved_sections if x}
    relevant = {x for x in relevant_sections if x}

    if not retrieved:
        return 0.0

    return len(retrieved & relevant) / len(retrieved)


def recall_at_k(retrieved_sections: list, relevant_sections: list) -> float:
    retrieved = {x for x in retrieved_sections if x}
    relevant = {x for x in relevant_sections if x}

    if not relevant:
        return 0.0

    return len(retrieved & relevant) / len(relevant)
