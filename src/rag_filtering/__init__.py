"""RAG answer filtering thesis package."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["RAGConfig", "RAGSystem"]

if TYPE_CHECKING:
    from rag_filtering.rag.config import RAGConfig
    from rag_filtering.rag.rag_system import RAGSystem


def __getattr__(name: str):
    if name == "RAGConfig":
        from rag_filtering.rag.config import RAGConfig

        return RAGConfig
    if name == "RAGSystem":
        from rag_filtering.rag.rag_system import RAGSystem

        return RAGSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
