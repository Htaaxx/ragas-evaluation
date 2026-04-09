"""
Backward-compatible re-export shim.

The original monolithic ``ragas_reward_filter.py`` has been split into:
  - data_models.py   — dataclasses, literature priors
  - metrics.py       — HybridMetricBundle
  - weight_fitting.py — WeightBank, WeightFitter
  - reward_filter.py — RAGASRewardComputer, ThresholdCalibrator, RAGASRewardFilter

This file re-exports every public name so that existing ``from
src.filtering.ragas_reward_filter import X`` statements continue to work.
"""

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
    "RAGASReward",
    "FilterDiagnostics",
    "CalibrationRecord",
    "LITERATURE_PRIORS",
    "HybridMetricBundle",
    "WeightBank",
    "WeightFitter",
    "RAGASRewardComputer",
    "ThresholdCalibrator",
    "RAGASRewardFilter",
]
