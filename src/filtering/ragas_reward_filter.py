"""
Backward-compatible re-export shim.

The filtering pipeline has been rewritten for black-box answer scoring.
This file re-exports every public name so that existing ``from
src.filtering.ragas_reward_filter import X`` statements continue to work.
"""

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
    "AnswerReward",
    "FilterDiagnostics",
    "ANSWER_WEIGHT_PRIORS",
    "AnswerMetricBundle",
    "WeightBank",
    "WeightFitter",
    "AnswerRewardComputer",
    "AnswerRewardFilter",
]
