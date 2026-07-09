"""
Filtering module for RAG quality improvement.

Provides:
- AnswerQualityClassifier — learned accept/reject filter (no ground truth needed)
- NLIAnswerFilter — zero-shot NLI-based answer filter (no training needed)
- FilterEvaluator / FilterResult — evaluation harness with 6 required metrics
- FilterDecision — structured accept/reject + confidence output
- AnswerFilter — LLM-as-judge answer scoring vs ground truth
- AnswerRewardFilter — generate-then-score pipeline
- AnswerMetricBundle — RAGAS + lexical answer-correctness metrics
"""

from .data_models import (
    ANSWER_WEIGHT_PRIORS,
    AnswerReward,
    FilterDecision,
    FilterDiagnostics,
)
from .ensemble_filter import EnsembleFilter
from .filter_evaluator import FilterEvaluator, FilterResult
from .learned_filter import AnswerQualityClassifier, train_classifier
from .llm_filter import AnswerFilter, AnswerScoreResult
from .metrics import AnswerMetricBundle
from .nli_filter import NLIAnswerFilter
from .ragas_feature_extractor import RagasFeatureExtractor, build_ragas_features
from .ragas_filter import RagasFilter, run_ragas_filter
from .ragas_filter_trainer import RagasFilterTrainer, train_ragas_filter
from .reward_filter import (
    AnswerRewardComputer,
    AnswerRewardFilter,
)
from .weight_fitting import WeightBank, WeightFitter

__all__ = [
    # Learned filter (core thesis)
    "AnswerQualityClassifier",
    "train_classifier",
    # NLI zero-shot filter
    "NLIAnswerFilter",
    # RAGAS-feature filter (feature extractor -> trainer -> filter)
    "RagasFeatureExtractor",
    "build_ragas_features",
    "RagasFilterTrainer",
    "train_ragas_filter",
    "RagasFilter",
    "run_ragas_filter",
    # Ensemble filter
    "EnsembleFilter",
    "FilterDecision",
    "FilterEvaluator",
    "FilterResult",
    # LLM answer scoring
    "AnswerFilter",
    "AnswerScoreResult",
    # Reward pipeline
    "AnswerRewardFilter",
    "AnswerRewardComputer",
    # Metric bundle
    "AnswerMetricBundle",
    # Weight components
    "WeightFitter",
    "WeightBank",
    "ANSWER_WEIGHT_PRIORS",
    # Data containers
    "AnswerReward",
    "FilterDiagnostics",
]
