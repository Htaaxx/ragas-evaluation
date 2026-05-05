"""
Data containers for the black-box answer filtering pipeline.

These dataclasses carry per-sample scoring results and diagnostics
through the answer evaluation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AnswerReward:
    """Per-sample answer-correctness reward (scored against ground truth)."""

    answer_correctness: float = 0.0
    answer_similarity: float = 0.0
    token_f1: float = 0.0
    rouge_l: float = 0.0
    composite: float = 0.0
    mode: str = "full"  # "full" | "lexical_only" | "skipped"

    def to_dict(self) -> Dict[str, float]:
        return {
            "answer_correctness": self.answer_correctness,
            "answer_similarity": self.answer_similarity,
            "token_f1": self.token_f1,
            "rouge_l": self.rouge_l,
            "composite": self.composite,
            "mode": self.mode,
        }


@dataclass
class FilterDecision:
    """Accept/reject decision from the learned answer quality filter."""

    accept: bool
    confidence: float
    reasoning: str


@dataclass
class FilterDiagnostics:
    """Diagnostic information emitted by the answer filter for one sample."""

    question: str
    ground_truth: str
    generated_answer: str
    correctness_score: float
    reward: AnswerReward
    full_metrics: Dict[str, float] = field(default_factory=dict)
    accepted: bool = False
    retries: int = 0


# ---------------------------------------------------------------------------
# Answer-correctness-centric weight priors per dataset type
# ---------------------------------------------------------------------------
ANSWER_WEIGHT_PRIORS: Dict[str, Dict[str, float]] = {
    "asqa": {
        "answer_correctness": 0.30,
        "answer_similarity": 0.25,
        "token_f1": 0.25,
        "rouge_l": 0.20,
    },
    "hotpotqa": {
        "answer_correctness": 0.30,
        "answer_similarity": 0.20,
        "token_f1": 0.30,
        "rouge_l": 0.20,
    },
    "factoid": {
        "answer_correctness": 0.35,
        "answer_similarity": 0.25,
        "token_f1": 0.25,
        "rouge_l": 0.15,
    },
    "universal": {
        "answer_correctness": 0.30,
        "answer_similarity": 0.25,
        "token_f1": 0.25,
        "rouge_l": 0.20,
    },
}
