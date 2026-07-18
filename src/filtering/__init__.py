"""
Filtering module for RAG answer-quality verification.

Heavy dependencies (``torch``, ``transformers``) are imported lazily so
light helpers like ``config_loader`` / ``data_split`` work on Kaggle without
triggering a half-initialized ``torch`` import during package load.
"""

from __future__ import annotations

from typing import Any

# Light imports only — no torch / transformers.
from .data_models import FilterDecision
from .data_split import load_and_split, to_base_id
from .deberta_filter_evaluator import FilterEvaluator as DebertaFilterEvaluator
from .deberta_filter_evaluator import (
    average_classification_reports,
    classification_report_by_dataset,
    select_threshold_min_fpr,
)

__all__ = [
    "AnswerQualityClassifier",
    "DebertaFilterEvaluator",
    "FilterDecision",
    "LLMJudgeFilter",
    "NLIAnswerFilter",
    "RAGAS",
    "RagasFeatureExtractor",
    "RagasFilter",
    "RagasFilterTrainer",
    "average_classification_reports",
    "build_ragas_features",
    "classification_report_by_dataset",
    "load_and_split",
    "run_ragas_filter",
    "select_threshold_min_fpr",
    "to_base_id",
    "train_classifier",
    "train_ragas_filter",
]

_LAZY_ATTRS = {
    "AnswerQualityClassifier": (".learned_filter", "AnswerQualityClassifier"),
    "train_classifier": (".learned_filter", "train_classifier"),
    "NLIAnswerFilter": (".nli_filter", "NLIAnswerFilter"),
    "LLMJudgeFilter": (".llm_judge_filter", "LLMJudgeFilter"),
    "RAGAS": (".ragas", "RAGAS"),
    "RagasFeatureExtractor": (".ragas_feature_extractor", "RagasFeatureExtractor"),
    "build_ragas_features": (".ragas_feature_extractor", "build_ragas_features"),
    "RagasFilter": (".ragas_filter", "RagasFilter"),
    "run_ragas_filter": (".ragas_filter", "run_ragas_filter"),
    "RagasFilterTrainer": (".ragas_filter_trainer", "RagasFilterTrainer"),
    "train_ragas_filter": (".ragas_filter_trainer", "train_ragas_filter"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    import importlib

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))
