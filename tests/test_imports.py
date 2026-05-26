"""Import sanity checks."""

from __future__ import annotations


def test_package_imports() -> None:
    from rag_filtering.filtering.learned_filter import (  # noqa: F401
        AnswerQualityClassifier,
        train_classifier,
    )
    from rag_filtering.filtering.data_split import load_and_split  # noqa: F401
    from rag_filtering.config.loader import load_filtering_config  # noqa: F401


def test_rag_imports() -> None:
    from rag_filtering.rag.config import RAGConfig  # noqa: F401
    from rag_filtering.rag.rag_system import RAGSystem  # noqa: F401
