"""
RAG Training System

A comprehensive framework for training and fine-tuning RAG models.
"""

from .helper import (
    DEFAULT_RAGAS_FEATURE_COLS,
    GOLD_ANSWER_CANDIDATES,
    _ensure_path,
    _safe_json_dump,
    _normalize_col_aliases,
    parse_context,
    _get_gold_col,
    get_default_models,
)

__version__ = "1.0.0"

__all__ = [
    "RAGConfig",
    "RAGSystem",
    "DEFAULT_RAGAS_FEATURE_COLS",
    "GOLD_ANSWER_CANDIDATES",
    "_ensure_path",
    "_safe_json_dump",
    "_normalize_col_aliases",
    "parse_context",
    "_get_gold_col",
    "get_default_models",
]
