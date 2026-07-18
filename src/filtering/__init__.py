"""
Filtering module for RAG answer-quality verification.

Provides:
- AnswerQualityClassifier — fine-tuned DeBERTa faithfulness filter
- NLIAnswerFilter — zero-shot NLI baseline
- RagasFilter / trainers — RAGAS-feature filter (primary thesis method)
- LLMJudgeFilter — LLM-as-judge baseline
- load_and_split — leakage-safe train/val/test split by base question ID
- FilterDecision — structured accept/reject + confidence output
"""

from .data_models import FilterDecision
from .data_split import load_and_split, to_base_id
from .deberta_filter_evaluator import FilterEvaluator as DebertaFilterEvaluator
from .deberta_filter_evaluator import (
    average_classification_reports,
    classification_report_by_dataset,
    select_threshold_min_fpr,
)
from .learned_filter import AnswerQualityClassifier, train_classifier
from .llm_judge_filter import LLMJudgeFilter
from .nli_filter import NLIAnswerFilter
from .ragas import RAGAS
from .ragas_feature_extractor import RagasFeatureExtractor, build_ragas_features
from .ragas_filter import RagasFilter, run_ragas_filter
from .ragas_filter_trainer import RagasFilterTrainer, train_ragas_filter

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
