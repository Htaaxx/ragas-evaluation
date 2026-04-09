"""
H-RRGF — Hybrid RAGAS-Reward-Guided Filtering.

Orchestrates RAGASRewardComputer, ThresholdCalibrator, and
RAGASRewardFilter for reward-guided context filtering with
CRAG-inspired retry loops.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .data_models import (
    CalibrationRecord,
    FilterDiagnostics,
    RAGASReward,
)
from .llm_filter import ContextFilter, ContextFilterResult, LLMFilterPipeline
from .metrics import HybridMetricBundle
from .weight_fitting import WeightBank, WeightFitter

logger = logging.getLogger(__name__)

try:
    from ragas import evaluate as ragas_evaluate  # noqa: F401

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# RAGASRewardComputer
# ---------------------------------------------------------------------------


class RAGASRewardComputer:
    """
    Computes the composite RAGAS reward using HybridMetricBundle and WeightBank.

    Exposes ``compute_proxy()`` and ``compute_full()`` returning
    ``List[RAGASReward]``.
    """

    def __init__(
        self,
        weight_bank: Optional[WeightBank] = None,
        dataset_type: str = "universal",
        llm_model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
    ) -> None:
        self.weight_bank = weight_bank or WeightBank()
        self.dataset_type = dataset_type
        self.metric_bundle = HybridMetricBundle()
        self.llm_model = llm_model
        self.embedding_model = embedding_model

    @property
    def weights(self) -> Dict[str, float]:
        return self.weight_bank.get_weights(self.dataset_type)

    def _composite(self, scores: Dict[str, float]) -> float:
        w = self.weights
        return sum(w.get(k, 0.0) * scores.get(k, 0.0) for k in w)

    def compute_proxy(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
    ) -> List[RAGASReward]:
        """Reference-free reward (3 metrics). Safe at inference time."""
        scores_list = self.metric_bundle.compute_proxy(questions, answers, contexts)
        return [
            RAGASReward(
                faithfulness=s.get("faithfulness", 0.0),
                answer_relevancy=s.get("answer_relevancy", 0.0),
                composite=self._composite(s),
                mode="proxy",
            )
            for s in scores_list
        ]

    def compute_full(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> List[RAGASReward]:
        """Full reward (up to 9 metrics). Requires ground truth."""
        scores_list = self.metric_bundle.compute_full(
            questions, answers, contexts, ground_truths
        )
        return [
            RAGASReward(
                faithfulness=s.get("faithfulness", 0.0),
                answer_relevancy=s.get("answer_relevancy", 0.0),
                context_precision=s.get("context_precision", 0.0),
                context_recall=s.get("context_recall", 0.0),
                composite=self._composite(s),
                mode="full",
            )
            for s in scores_list
        ]

    def scalar(self, reward: RAGASReward) -> float:
        return reward.composite


# ---------------------------------------------------------------------------
# ThresholdCalibrator
# ---------------------------------------------------------------------------


class ThresholdCalibrator:
    """Calibrates context-filter threshold τ* on a labelled split."""

    CANDIDATE_THRESHOLDS: List[float] = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    def __init__(
        self,
        context_filter: ContextFilter,
        reward_computer: RAGASRewardComputer,
        generation_fn: Callable[[str, List[str]], str],
        weight_fitter: Optional[WeightFitter] = None,
        weight_method: str = "correlation",
        save_path: Optional[Path] = None,
    ) -> None:
        self.context_filter = context_filter
        self.reward_computer = reward_computer
        self.generation_fn = generation_fn
        self.weight_fitter = weight_fitter or WeightFitter()
        self.weight_method = weight_method
        self.save_path = save_path
        self.records: List[CalibrationRecord] = []
        self.optimal_threshold: float = 6.0

    def calibrate(
        self,
        questions: List[str],
        passages_list: List[List[str]],
        ground_truths: List[str],
        dataset_type: str = "universal",
        verbose: bool = True,
    ) -> float:
        """Run full calibration and return τ*."""
        bundle = self.reward_computer.metric_bundle

        # Step 0 — Score passages
        if verbose:
            logger.info("Step 0 — Scoring passages via ContextFilter …")
        all_scores, all_passages = self._score_all_passages(
            questions, passages_list
        )

        # Step 1 — Baseline generation
        if verbose:
            logger.info("Step 1 — Generating baseline answers (unfiltered) …")
        baseline_answers = [
            self.generation_fn(q, ctx)
            for q, ctx in zip(questions, all_passages)
        ]

        # Step 2 — Full metric bundle
        if verbose:
            logger.info("Step 2 — Computing full metric bundle …")
        full_metrics = bundle.compute_full(
            questions, baseline_answers, all_passages, ground_truths
        )

        # Step 3 — Fit weights
        downstream_y = [m.get("token_f1", 0.0) for m in full_metrics]
        fitted_weights = self.weight_fitter.fit(
            full_metrics, downstream_y, method=self.weight_method
        )
        if verbose:
            logger.info("Step 3 — Fitted weights (%s):", self.weight_method)
            for k, v in sorted(fitted_weights.items(), key=lambda x: -x[1]):
                logger.info("    %s: %.4f", k, v)

        # Step 4 — Update WeightBank
        self.reward_computer.weight_bank.update(dataset_type, fitted_weights)
        self.reward_computer.dataset_type = dataset_type

        # Step 5 — Threshold sweep
        if verbose:
            logger.info("Step 5 — Threshold sweep …")
        best_tau, best_reward, records = self._sweep_thresholds(
            questions, all_scores, all_passages, ground_truths,
            fitted_weights, bundle, verbose,
        )

        self.records = records
        self.optimal_threshold = best_tau

        if verbose:
            logger.info("τ* = %.1f  (mean composite = %.4f)", best_tau, best_reward)

        # Step 6 — Persist
        if self.save_path:
            self._save(records, best_tau, fitted_weights)

        return best_tau

    def load(self) -> Optional[float]:
        """Load previously saved calibration. Returns τ* or None."""
        if not self.save_path or not Path(self.save_path).exists():
            return None
        with open(self.save_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.optimal_threshold = data["optimal_threshold"]
        self.records = [
            CalibrationRecord(
                **{k: v for k, v in r.items()
                   if k in CalibrationRecord.__dataclass_fields__}
            )
            for r in data["records"]
        ]
        if "fitted_weights" in data and "dataset_type" in data:
            self.reward_computer.weight_bank.update(
                data["dataset_type"], data["fitted_weights"]
            )
            self.reward_computer.dataset_type = data["dataset_type"]
        return self.optimal_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_all_passages(
        self,
        questions: List[str],
        passages_list: List[List[str]],
    ) -> Tuple[List[List[float]], List[List[str]]]:
        all_scores: List[List[float]] = []
        all_passages: List[List[str]] = []
        for q, passages in zip(questions, passages_list):
            results = self.context_filter.filter_contexts(q, passages)
            all_scores.append([r.score for r in results])
            all_passages.append([r.passage for r in results])
        return all_scores, all_passages

    def _sweep_thresholds(
        self,
        questions: List[str],
        all_scores: List[List[float]],
        all_passages: List[List[str]],
        ground_truths: List[str],
        fitted_weights: Dict[str, float],
        bundle: HybridMetricBundle,
        verbose: bool,
    ) -> Tuple[float, float, List[CalibrationRecord]]:
        records: List[CalibrationRecord] = []
        best_tau, best_reward = self.CANDIDATE_THRESHOLDS[0], -1.0

        for tau in self.CANDIDATE_THRESHOLDS:
            filtered_contexts = self._apply_threshold(all_scores, all_passages, tau)
            answers = [
                self.generation_fn(q, ctx)
                for q, ctx in zip(questions, filtered_contexts)
            ]
            metrics = bundle.compute_full(
                questions, answers, filtered_contexts, ground_truths
            )
            composites = [
                sum(fitted_weights.get(k, 0.0) * m.get(k, 0.0) for k in fitted_weights)
                for m in metrics
            ]
            mean_composite = float(np.mean(composites))

            rec = CalibrationRecord(
                threshold=tau,
                mean_composite=mean_composite,
                mean_faithfulness=float(np.mean([m.get("faithfulness", 0.0) for m in metrics])),
                mean_answer_relevancy=float(np.mean([m.get("answer_relevancy", 0.0) for m in metrics])),
                mean_context_precision=float(np.mean([m.get("context_precision", 0.0) for m in metrics])),
                mean_context_recall=float(np.mean([m.get("context_recall", 0.0) for m in metrics])),
                n_samples=len(questions),
                fitted_weights=fitted_weights,
            )
            records.append(rec)

            if verbose:
                logger.info(
                    "  τ=%.1f  composite=%.4f  faith=%.4f  ans_rel=%.4f",
                    tau, mean_composite, rec.mean_faithfulness,
                    rec.mean_answer_relevancy,
                )

            if mean_composite > best_reward or (
                mean_composite == best_reward and tau > best_tau
            ):
                best_reward = mean_composite
                best_tau = tau

        return best_tau, best_reward, records

    @staticmethod
    def _apply_threshold(
        all_scores: List[List[float]],
        all_passages: List[List[str]],
        tau: float,
    ) -> List[List[str]]:
        result: List[List[str]] = []
        for scores, passages in zip(all_scores, all_passages):
            kept = [p for p, s in zip(passages, scores) if s >= tau]
            if not kept:
                best_idx = int(np.argmax(scores)) if scores else 0
                kept = [passages[best_idx]] if passages else [""]
            result.append(kept)
        return result

    def _save(
        self,
        records: List[CalibrationRecord],
        optimal: float,
        fitted_weights: Dict[str, float],
    ) -> None:
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "optimal_threshold": optimal,
            "fitted_weights": fitted_weights,
            "dataset_type": self.reward_computer.dataset_type,
            "weight_fitting_method": self.weight_method,
            "records": [
                {
                    "threshold": r.threshold,
                    "mean_composite": r.mean_composite,
                    "mean_faithfulness": r.mean_faithfulness,
                    "mean_answer_relevancy": r.mean_answer_relevancy,
                    "mean_context_precision": r.mean_context_precision,
                    "mean_context_recall": r.mean_context_recall,
                    "n_samples": r.n_samples,
                    "fitted_weights": r.fitted_weights,
                }
                for r in records
            ],
        }
        with open(self.save_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Calibration saved -> %s", self.save_path)


# ---------------------------------------------------------------------------
# RAGASRewardFilter
# ---------------------------------------------------------------------------


class RAGASRewardFilter:
    """H-RRGF — public interface for reward-guided filtering at inference."""

    def __init__(
        self,
        llm_filter: LLMFilterPipeline,
        generation_fn: Callable[[str, List[str]], str],
        reward_computer: Optional[RAGASRewardComputer] = None,
        weight_bank: Optional[WeightBank] = None,
        dataset_type: str = "universal",
        optimal_threshold: float = 6.0,
        min_passages: int = 1,
        max_passages: int = 5,
        min_reward: float = 0.50,
        retry_delta: float = 1.0,
        max_retries: int = 3,
        use_proxy_reward: bool = True,
        weight_fitting_method: str = "correlation",
    ) -> None:
        self.llm_filter = llm_filter
        self.generation_fn = generation_fn
        self.optimal_threshold = optimal_threshold
        self.min_passages = min_passages
        self.max_passages = max_passages
        self.min_reward = min_reward
        self.retry_delta = retry_delta
        self.max_retries = max_retries
        self.use_proxy_reward = use_proxy_reward
        self.weight_fitting_method = weight_fitting_method
        self.dataset_type = dataset_type

        if reward_computer is not None:
            self.reward_computer = reward_computer
        else:
            wb = weight_bank or WeightBank()
            self.reward_computer = RAGASRewardComputer(
                weight_bank=wb, dataset_type=dataset_type,
            )

        if use_proxy_reward and not _RAGAS_AVAILABLE:
            logger.warning("RAGAS not installed — proxy reward disabled.")
            self.use_proxy_reward = False

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        passages: List[str],
    ) -> Tuple[str, List[str], FilterDiagnostics]:
        """Run H-RRGF for a single question with CRAG retry loop."""
        raw_results: List[ContextFilterResult] = (
            self.llm_filter.context_filter.filter_contexts(question, passages)
        )
        scores = [r.score for r in raw_results]
        scored = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
        scored = scored[: self.max_passages]

        retries = 0
        tau = self.optimal_threshold
        best_answer = ""
        best_contexts: List[str] = []
        best_reward = RAGASReward()
        best_full_metrics: Dict[str, float] = {}
        fallback_used = False

        while retries <= self.max_retries:
            filtered = [p for s, p in scored if s >= tau]
            if len(filtered) < self.min_passages:
                filtered = [p for _, p in scored[: self.min_passages]]
                fallback_used = True

            gen_answer = self.generation_fn(question, filtered)

            if self.use_proxy_reward:
                proxy_scores = self.reward_computer.metric_bundle.compute_proxy(
                    [question], [gen_answer], [filtered]
                )
                reward = self.reward_computer.compute_proxy(
                    [question], [gen_answer], [filtered]
                )[0]
                full_m = proxy_scores[0]
            else:
                reward = RAGASReward(composite=1.0, mode="skipped")
                full_m = {}

            if retries == 0 or reward.composite > best_reward.composite:
                best_answer = gen_answer
                best_contexts = filtered
                best_reward = reward
                best_full_metrics = full_m

            if reward.composite >= self.min_reward or not self.use_proxy_reward:
                break

            tau = max(0.0, tau - self.retry_delta)
            retries += 1

        return best_answer, best_contexts, FilterDiagnostics(
            question=question,
            passages_before=len(passages),
            passages_after=len(best_contexts),
            threshold_used=tau,
            passage_scores=scores,
            reward=best_reward,
            full_metrics=best_full_metrics,
            retries=retries,
            fallback_used=fallback_used,
        )

    def answer_batch(
        self,
        questions: List[str],
        passages_list: List[List[str]],
        show_progress: bool = True,
    ) -> List[Tuple[str, List[str], FilterDiagnostics]]:
        """Run H-RRGF on a batch."""
        iterator: Any = enumerate(zip(questions, passages_list))
        if show_progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(
                    enumerate(zip(questions, passages_list)),
                    total=len(questions),
                    desc="H-RRGF",
                )
            except ImportError:
                pass
        results: List[Tuple[str, List[str], FilterDiagnostics]] = []
        for _, (q, p) in iterator:
            results.append(self.answer(q, p))
        return results

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        questions: List[str],
        passages_list: List[List[str]],
        ground_truths: List[str],
        dataset_type: Optional[str] = None,
        save_path: Optional[Path] = None,
        verbose: bool = True,
    ) -> float:
        """Calibrate threshold and fit weights on a labelled split."""
        if not _RAGAS_AVAILABLE:
            raise RuntimeError("RAGAS is required for calibration.")

        dt = dataset_type or self.dataset_type
        calibrator = ThresholdCalibrator(
            context_filter=self.llm_filter.context_filter,
            reward_computer=self.reward_computer,
            generation_fn=self.generation_fn,
            weight_fitter=WeightFitter(),
            weight_method=self.weight_fitting_method,
            save_path=save_path,
        )

        if save_path and Path(save_path).exists():
            cached_tau = calibrator.load()
            if cached_tau is not None:
                if verbose:
                    logger.info("Loaded cached calibration: τ* = %.1f", cached_tau)
                self.optimal_threshold = cached_tau
                return cached_tau

        tau_star = calibrator.calibrate(
            questions, passages_list, ground_truths,
            dataset_type=dt, verbose=verbose,
        )
        self.optimal_threshold = tau_star
        return tau_star

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
            "avg_passages_before": float(np.mean([d.passages_before for d in diagnostics])),
            "avg_passages_after": float(np.mean([d.passages_after for d in diagnostics])),
            "avg_reduction_pct": float(np.mean([
                (1 - d.passages_after / max(d.passages_before, 1)) * 100
                for d in diagnostics
            ])),
            "avg_composite_reward": float(np.mean([d.reward.composite for d in diagnostics])),
            "avg_faithfulness": float(np.mean([d.reward.faithfulness for d in diagnostics])),
            "avg_answer_relevancy": float(np.mean([d.reward.answer_relevancy for d in diagnostics])),
            "pct_retried": float(np.mean([d.retries > 0 for d in diagnostics])) * 100,
            "pct_fallback": float(np.mean([d.fallback_used for d in diagnostics])) * 100,
            "avg_retries": float(np.mean([d.retries for d in diagnostics])),
        }
