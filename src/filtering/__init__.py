"""
Filtering module for RAG quality improvement.

Provides:
- BaseFilter — abstract interface for all filtering strategies
- LLM-as-judge context and answer filtering (llm_filter)
- H-RRGF — Hybrid RAGAS-Reward-Guided Filtering (reward_filter)
"""

from .base_filter import BaseFilter, FilterResult, ScoredDocument
from .llm_filter import AnswerFilter, ContextFilter, LLMFilterPipeline
from .data_models import (
    CalibrationRecord,
    FilterDiagnostics,
    LITERATURE_PRIORS,
    RAGASReward,
)
from .metrics import HybridMetricBundle
from .reward_filter import (
    RAGASRewardComputer,
    RAGASRewardFilter,
    ThresholdCalibrator,
)
from .weight_fitting import WeightBank, WeightFitter

__all__ = [
    # Base interface
    "BaseFilter",
    "FilterResult",
    "ScoredDocument",
    # LLM filter
    "ContextFilter",
    "AnswerFilter",
    "LLMFilterPipeline",
    # H-RRGF pipeline
    "RAGASRewardFilter",
    "RAGASRewardComputer",
    "ThresholdCalibrator",
    # Hybrid metric / weight components
    "HybridMetricBundle",
    "WeightFitter",
    "WeightBank",
    "LITERATURE_PRIORS",
    # Data containers
    "RAGASReward",
    "FilterDiagnostics",
    "CalibrationRecord",
]
