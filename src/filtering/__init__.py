"""
Filtering module for RAG quality improvement.

Black-box answer-only filtering: score generated answers against
ground truth.  No context filtering.

Provides:
- AnswerFilter — LLM-as-judge answer scoring vs ground truth
- AnswerRewardFilter — generate-then-score pipeline
- AnswerMetricBundle — RAGAS + lexical answer-correctness metrics
"""

from .llm_filter import AnswerFilter, AnswerScoreResult
from .data_models import (
    ANSWER_WEIGHT_PRIORS,
    AnswerReward,
    FilterDiagnostics,
)
from .metrics import AnswerMetricBundle
from .reward_filter import (
    AnswerRewardComputer,
    AnswerRewardFilter,
)
from .weight_fitting import WeightBank, WeightFitter

__all__ = [
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
