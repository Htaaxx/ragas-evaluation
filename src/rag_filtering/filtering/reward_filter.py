"""
Black-box answer reward filter.

Scores generated answers against ground truth using AnswerMetricBundle
and WeightBank.  No context filtering — all retrieved passages go
straight to the generator; only the final answer is evaluated.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .data_models import AnswerReward, FilterDiagnostics
from .metrics import AnswerMetricBundle
from .weight_fitting import WeightBank

logger = logging.getLogger(__name__)

try:
    from ragas import evaluate as ragas_evaluate  # noqa: F401

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# AnswerRewardComputer
# ---------------------------------------------------------------------------


class AnswerRewardComputer:
    """
    Computes the composite answer-correctness reward.

    Uses AnswerMetricBundle to obtain per-sample metrics and WeightBank
    to produce a single composite scalar.
    """

    def __init__(
        self,
        weight_bank: Optional[WeightBank] = None,
        dataset_type: str = "universal",
    ) -> None:
        self.weight_bank = weight_bank or WeightBank()
        self.dataset_type = dataset_type
        self.metric_bundle = AnswerMetricBundle()

    @property
    def weights(self) -> Dict[str, float]:
        return self.weight_bank.get_weights(self.dataset_type)

    def _composite(self, scores: Dict[str, float]) -> float:
        w = self.weights
        return sum(w.get(k, 0.0) * scores.get(k, 0.0) for k in w)

    def compute(
        self,
        questions: List[str],
        answers: List[str],
        ground_truths: List[str],
    ) -> List[AnswerReward]:
        """Score answers against ground truth and return per-sample rewards."""
        scores_list = self.metric_bundle.compute(questions, answers, ground_truths)
        return [
            AnswerReward(
                answer_correctness=s.get("answer_correctness", 0.0),
                answer_similarity=s.get("answer_similarity", 0.0),
                token_f1=s.get("token_f1", 0.0),
                rouge_l=s.get("rouge_l", 0.0),
                composite=self._composite(s),
                mode="full" if _RAGAS_AVAILABLE else "lexical_only",
            )
            for s in scores_list
        ]

    def scalar(self, reward: AnswerReward) -> float:
        return reward.composite


# ---------------------------------------------------------------------------
# AnswerRewardFilter
# ---------------------------------------------------------------------------


class AnswerRewardFilter:
    """
    Black-box answer filter: retrieve -> generate -> score -> retry.

    All retrieved passages go to the generator unfiltered.  The generated
    answer is scored against ground truth.  If the score falls below
    ``correctness_threshold``, the generator is called again (up to
    ``max_retries`` times) and the best answer across all attempts is kept.
    """

    def __init__(
        self,
        generation_fn: Callable[[str, List[str]], str],
        reward_computer: Optional[AnswerRewardComputer] = None,
        weight_bank: Optional[WeightBank] = None,
        dataset_type: str = "universal",
        correctness_threshold: float = 0.50,
        max_retries: int = 3,
    ) -> None:
        self.generation_fn = generation_fn
        self.correctness_threshold = correctness_threshold
        self.max_retries = max_retries
        self.dataset_type = dataset_type

        if reward_computer is not None:
            self.reward_computer = reward_computer
        else:
            wb = weight_bank or WeightBank()
            self.reward_computer = AnswerRewardComputer(
                weight_bank=wb, dataset_type=dataset_type,
            )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        passages: List[str],
        ground_truth: str,
    ) -> Tuple[str, FilterDiagnostics]:
        """Generate, score, and optionally retry to find a better answer."""
        best_answer = ""
        best_reward = AnswerReward()
        retries = 0

        while retries <= self.max_retries:
            gen_answer = self.generation_fn(question, passages)

            reward = self.reward_computer.compute(
                [question], [gen_answer], [ground_truth]
            )[0]

            if retries == 0 or reward.composite > best_reward.composite:
                best_answer = gen_answer
                best_reward = reward

            if best_reward.composite >= self.correctness_threshold:
                break

            retries += 1

        accepted = best_reward.composite >= self.correctness_threshold

        diagnostics = FilterDiagnostics(
            question=question,
            ground_truth=ground_truth,
            generated_answer=best_answer,
            correctness_score=best_reward.composite,
            reward=best_reward,
            full_metrics=best_reward.to_dict(),
            accepted=accepted,
            retries=retries,
        )

        return best_answer, diagnostics

    def answer_batch(
        self,
        questions: List[str],
        passages_list: List[List[str]],
        ground_truths: List[str],
        show_progress: bool = True,
    ) -> List[Tuple[str, FilterDiagnostics]]:
        """Generate, score, and retry a batch of answers."""
        iterator: Any = enumerate(zip(questions, passages_list, ground_truths))
        if show_progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(
                    enumerate(zip(questions, passages_list, ground_truths)),
                    total=len(questions),
                    desc="Answer scoring",
                )
            except ImportError:
                pass

        results: List[Tuple[str, FilterDiagnostics]] = []
        for _, (q, p, gt) in iterator:
            results.append(self.answer(q, p, gt))
        return results

    # ------------------------------------------------------------------
    # Diagnostics summary
    # ------------------------------------------------------------------

    @staticmethod
    def summarise_diagnostics(
        diagnostics: List[FilterDiagnostics],
    ) -> Dict[str, Any]:
        """Aggregate diagnostic statistics across a batch."""
        n = len(diagnostics)
        if n == 0:
            return {}
        return {
            "n_samples": n,
            "avg_composite_reward": float(np.mean([d.reward.composite for d in diagnostics])),
            "avg_answer_correctness": float(np.mean([d.reward.answer_correctness for d in diagnostics])),
            "avg_answer_similarity": float(np.mean([d.reward.answer_similarity for d in diagnostics])),
            "avg_token_f1": float(np.mean([d.reward.token_f1 for d in diagnostics])),
            "avg_rouge_l": float(np.mean([d.reward.rouge_l for d in diagnostics])),
            "pct_accepted": float(np.mean([d.accepted for d in diagnostics])) * 100,
            "pct_retried": float(np.mean([d.retries > 0 for d in diagnostics])) * 100,
            "avg_retries": float(np.mean([d.retries for d in diagnostics])),
        }
