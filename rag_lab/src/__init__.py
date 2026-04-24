"""
QBrain RAG mini-lab — layered package.

Import concrete modules explicitly, e.g.:

    from services.rag_service import RAGService
    from application.chunking import chunk_text

Keeping this file free of heavy re-exports avoids pulling optional stacks when
notebooks only need one submodule (and matches clearer dependency boundaries).
"""

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy `from qbrain_rag import RAGService` without eager submodule imports."""
    if name == "RAGService":
        from services.rag_service import RAGService

        return RAGService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["RAGService"]
