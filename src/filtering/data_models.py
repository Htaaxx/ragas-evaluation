"""
Data containers for the H-RRGF (Hybrid RAGAS-Reward-Guided Filtering) system.

These dataclasses carry per-sample results, diagnostics, and calibration
records through the filtering pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RAGASReward:
    """Per-sample composite RAGAS reward."""

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    composite: float = 0.0
    mode: str = "proxy"  # "proxy" | "full" | "skipped"

    def to_dict(self) -> Dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "composite": self.composite,
            "mode": self.mode,
        }


@dataclass
class FilterDiagnostics:
    """Diagnostic information emitted by H-RRGF for one sample."""

    question: str
    passages_before: int
    passages_after: int
    threshold_used: float
    passage_scores: List[float]
    reward: RAGASReward
    full_metrics: Dict[str, float] = field(default_factory=dict)
    retries: int = 0
    fallback_used: bool = False


@dataclass
class CalibrationRecord:
    """One calibration data point (threshold candidate -> mean reward)."""

    threshold: float
    mean_composite: float
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    mean_context_recall: float
    n_samples: int
    fitted_weights: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Literature-based weight priors per dataset type
# ---------------------------------------------------------------------------
LITERATURE_PRIORS: Dict[str, Dict[str, float]] = {
    "asqa": {
        "faithfulness": 0.20,
        "answer_relevancy": 0.18,
        "context_precision": 0.14,
        "context_recall": 0.14,
        "answer_correctness": 0.12,
        "answer_similarity": 0.08,
        "context_relevancy": 0.07,
        "token_f1": 0.05,
        "rouge_l": 0.02,
    },
    "hotpotqa": {
        "faithfulness": 0.18,
        "answer_relevancy": 0.14,
        "context_precision": 0.09,
        "context_recall": 0.28,
        "answer_correctness": 0.12,
        "answer_similarity": 0.07,
        "context_relevancy": 0.06,
        "token_f1": 0.04,
        "rouge_l": 0.02,
    },
    "factoid": {
        "faithfulness": 0.28,
        "answer_relevancy": 0.22,
        "context_precision": 0.17,
        "context_recall": 0.08,
        "answer_correctness": 0.11,
        "answer_similarity": 0.06,
        "context_relevancy": 0.04,
        "token_f1": 0.03,
        "rouge_l": 0.01,
    },
    "universal": {
        "faithfulness": 0.22,
        "answer_relevancy": 0.18,
        "context_precision": 0.13,
        "context_recall": 0.17,
        "answer_correctness": 0.12,
        "answer_similarity": 0.07,
        "context_relevancy": 0.06,
        "token_f1": 0.04,
        "rouge_l": 0.01,
    },
}
