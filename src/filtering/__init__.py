"""
LLM-based filtering module for RAG quality improvement.

This module provides:
- LLM-as-judge context and answer filtering (llm_filter)
- RAGAS-Reward-Guided Filtering (ragas_reward_filter) — uses RAGAS metrics
  as an explicit reward signal to calibrate and adaptively drive filtering
"""

from .llm_filter import ContextFilter, AnswerFilter, LLMFilterPipeline
from .ragas_reward_filter import (
    # Core pipeline
    RAGASRewardFilter,
    RAGASRewardComputer,
    ThresholdCalibrator,
    # Hybrid metric / weight components
    HybridMetricBundle,
    WeightFitter,
    WeightBank,
    LITERATURE_PRIORS,
    # Data containers
    RAGASReward,
    FilterDiagnostics,
    CalibrationRecord,
)

__all__ = [
    # Original LLM filter
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
