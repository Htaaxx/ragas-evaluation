"""
Hybrid RAGAS-Reward-Guided Filtering (H-RRGF) for RAG systems.

This module enhances the existing LLM-based filter by using a hybrid set of
RAGAS, lexical, and semantic metrics as an explicit reward signal to:
  1. Learn data-driven composite weights from calibration data (not hardcoded).
  2. Find the optimal context-filtering threshold via a reward-guided sweep.
  3. Generalise across datasets via a dataset-type-aware WeightBank.
  4. Correct low-quality answers at inference time via a CRAG-inspired retry loop.

Papers referenced
-----------------
1.  RAGAS (Es et al., 2023 — arXiv:2309.15217)
    Four core + three auxiliary metrics that together cover every axis of a
    RAG pipeline.  Table 3 is used as a weight *prior* seeded into WeightBank,
    not as a ground truth.

2.  CRAG — Corrective Retrieval Augmented Generation
    (Yan et al., 2024 — arXiv:2401.15884)
    Retrieval evaluator that decides USE / REFINE / DISCARD.  We generalise
    to a continuous composite reward + adaptive threshold retry loop.

3.  Self-RAG (Asai et al., 2023 — arXiv:2310.11511)
    Per-segment critique tokens emitted inline.  We mirror the philosophy
    (evaluate quality within the same inference pass) without fine-tuning.

4.  BERTScore (Zhang et al., 2020 — arXiv:1904.09675)
    Reference-based semantic similarity bridging lexical and LLM-based scores.

5.  ROUGE (Lin, 2004)
    Lexical overlap baseline — fast, free, strongly correlated with human
    judgement on long-form answers.

Composite reward
----------------
R = Σ_k  w_k · metric_k(q, C, a, [ground_truth])

Weights w_k are discovered via WeightFitter from calibration data; they are
NOT hardcoded.  WeightBank stores per-dataset-type priors as fallback.

Threshold calibration (offline, once per dataset split)
-------------------------------------------------------
Step 0  Score all passages once via ContextFilter (Gemini)
Step 1  Generate baseline answers (unfiltered) → compute full 9-metric bundle
Step 2  WeightFitter.fit(metric_scores, token_f1) → data-driven weights
Step 3  Sweep τ ∈ {3,4,5,6,7,8} using fitted weights → τ* = argmax mean R
Step 4  Persist τ*, weights, WeightBank to JSON

Threshold at inference (no calibration data available)
------------------------------------------------------
Start at τ=6.0 → generate → compute proxy reward (3 reference-free metrics)
If r̂ < ρ_min: lower τ by Δτ, regenerate (CRAG corrective loop, max 3 retries)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy dependencies — graceful degradation
# ---------------------------------------------------------------------------
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        answer_similarity,
        context_precision,
        context_recall,
        context_relevancy,
        faithfulness,
    )
    from datasets import Dataset as HFDataset
    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False

try:
    from bert_score import score as bert_score_fn
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False

try:
    from scipy.optimize import nnls as scipy_nnls
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from .llm_filter import ContextFilter, ContextFilterResult, LLMFilterPipeline


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RAGASReward:
    """Per-sample composite RAGAS reward (backward-compatible)."""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    composite: float = 0.0
    mode: str = "proxy"   # "proxy" | "full" | "skipped"

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
    """One calibration data point (threshold candidate → mean reward)."""
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
# These encode the structural requirements of each dataset type.
# When no calibration data is available, WeightBank serves these directly.
# When calibration data is available, WeightFitter replaces them.
#
# Metric ordering and rationale:
#   ASQA   : ambiguous long-form — both precision and recall matter equally;
#             faithfulness is slightly lower because long-form answers
#             legitimately synthesise across passages.
#   HotpotQA: multi-hop — context_recall dominates; missing any bridge
#             passage collapses the reasoning chain.
#   Factoid  : single-entity answer — faithfulness dominates; recall is low
#             priority because one precise passage is sufficient.
#   Universal: weighted harmonic mean across the three types above.

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


# ---------------------------------------------------------------------------
# HybridMetricBundle
# ---------------------------------------------------------------------------

class HybridMetricBundle:
    """
    Compute up to 9 quality metrics for a batch of (q, a, C, [gt]) triples.

    Proxy bundle (reference-free, inference-safe, 3 metrics):
        faithfulness, answer_relevancy, context_relevancy

    Full bundle (requires ground truth, calibration only, up to 9 metrics):
        + context_precision, context_recall, answer_correctness,
          answer_similarity, token_f1, rouge_l
        + bertscore_f1 if bert_score package is installed

    All RAGAS metrics are computed in a single ragas.evaluate() call per mode
    to minimise LLM round-trips.
    """

    PROXY_RAGAS = [faithfulness, answer_relevancy, context_relevancy] if _RAGAS_AVAILABLE else []
    FULL_RAGAS  = (
        [faithfulness, answer_relevancy, context_precision,
         context_recall, answer_correctness, answer_similarity,
         context_relevancy]
        if _RAGAS_AVAILABLE else []
    )

    # ------------------------------------------------------------------ #
    # Static lexical / semantic helpers                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def token_f1(pred: str, gold: str) -> float:
        """
        SQuAD-style token-level F1.  Case-insensitive, whitespace tokenised.
        Returns 0 if either string is empty.
        """
        pred_toks = pred.lower().split()
        gold_toks = gold.lower().split()
        if not pred_toks or not gold_toks:
            return 0.0
        common = set(pred_toks) & set(gold_toks)
        if not common:
            return 0.0
        prec = len(common) / len(pred_toks)
        rec  = len(common) / len(gold_toks)
        return 2.0 * prec * rec / (prec + rec)

    @staticmethod
    def rouge_l(pred: str, gold: str) -> float:
        """
        ROUGE-L F1 via dynamic programming on LCS.
        No external library required.
        """
        pred_toks = pred.lower().split()
        gold_toks = gold.lower().split()
        if not pred_toks or not gold_toks:
            return 0.0
        m, n = len(gold_toks), len(pred_toks)
        # LCS length via DP
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if gold_toks[i - 1] == pred_toks[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        prec = lcs / n
        rec  = lcs / m
        if prec + rec == 0:
            return 0.0
        return 2.0 * prec * rec / (prec + rec)

    # ------------------------------------------------------------------ #
    # Compute methods                                                       #
    # ------------------------------------------------------------------ #

    def compute_proxy(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
    ) -> List[Dict[str, float]]:
        """
        Reference-free metric bundle (3 metrics).
        Safe to call at inference time — no ground truth required.

        Returns one dict per sample with keys:
            faithfulness, answer_relevancy, context_relevancy
        """
        if not _RAGAS_AVAILABLE:
            return [{"faithfulness": 0.0, "answer_relevancy": 0.0,
                     "context_relevancy": 0.0} for _ in questions]

        dataset = HFDataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        })
        result_df = ragas_evaluate(dataset, metrics=self.PROXY_RAGAS).to_pandas()

        out = []
        for _, row in result_df.iterrows():
            out.append({
                "faithfulness":      float(row.get("faithfulness", 0.0)),
                "answer_relevancy":  float(row.get("answer_relevancy", 0.0)),
                "context_relevancy": float(row.get("context_relevancy", 0.0)),
            })
        return out

    def compute_full(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> List[Dict[str, float]]:
        """
        Full metric bundle (up to 9 metrics).
        Requires ground-truth answers — use during calibration only.

        Returns one dict per sample with keys:
            faithfulness, answer_relevancy, context_precision,
            context_recall, answer_correctness, answer_similarity,
            context_relevancy, token_f1, rouge_l
            [+ bertscore_f1 if bert_score installed]
        """
        if not _RAGAS_AVAILABLE:
            return self._lexical_only(answers, ground_truths)

        dataset = HFDataset.from_dict({
            "question":     questions,
            "answer":       answers,
            "contexts":     contexts,
            "ground_truth": ground_truths,
        })
        result_df = ragas_evaluate(dataset, metrics=self.FULL_RAGAS).to_pandas()

        out = []
        for i, (_, row) in enumerate(result_df.iterrows()):
            entry: Dict[str, float] = {
                "faithfulness":      float(row.get("faithfulness", 0.0)),
                "answer_relevancy":  float(row.get("answer_relevancy", 0.0)),
                "context_precision": float(row.get("context_precision", 0.0)),
                "context_recall":    float(row.get("context_recall", 0.0)),
                "answer_correctness":float(row.get("answer_correctness", 0.0)),
                "answer_similarity": float(row.get("answer_similarity", 0.0)),
                "context_relevancy": float(row.get("context_relevancy", 0.0)),
                # Lexical — computed directly from strings
                "token_f1":  self.token_f1(answers[i], ground_truths[i]),
                "rouge_l":   self.rouge_l(answers[i], ground_truths[i]),
            }
            out.append(entry)

        # Optional BERTScore
        if _BERTSCORE_AVAILABLE:
            try:
                _, _, f1s = bert_score_fn(answers, ground_truths, lang="en", verbose=False)
                for i, entry in enumerate(out):
                    entry["bertscore_f1"] = float(f1s[i])
            except Exception:
                pass

        return out

    def _lexical_only(
        self, answers: List[str], ground_truths: List[str]
    ) -> List[Dict[str, float]]:
        """Fallback when RAGAS is not installed — lexical metrics only."""
        return [
            {
                "faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0,
                "answer_correctness": 0.0, "answer_similarity": 0.0,
                "context_relevancy": 0.0,
                "token_f1": self.token_f1(a, g),
                "rouge_l":  self.rouge_l(a, g),
            }
            for a, g in zip(answers, ground_truths)
        ]


# ---------------------------------------------------------------------------
# WeightBank
# ---------------------------------------------------------------------------

class WeightBank:
    """
    Stores composite-reward weights per dataset type.

    Priority order when retrieving weights for a given dataset_type:
      1. Fitted weights (from WeightFitter.fit() on actual calibration data)
      2. Literature prior (LITERATURE_PRIORS — dataset-type-specific)
      3. Universal prior  (LITERATURE_PRIORS["universal"])

    JSON round-trip
    ---------------
    {
      "fitted":   { "asqa": {...}, "hotpotqa": {...} },  // data-driven
      "priors":   { "asqa": {...}, ... }                 // literature
    }
    """

    def __init__(self) -> None:
        # Deep copy priors so mutations don't bleed across instances
        self._priors: Dict[str, Dict[str, float]] = {
            k: dict(v) for k, v in LITERATURE_PRIORS.items()
        }
        self._fitted: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_weights(self, dataset_type: str = "universal") -> Dict[str, float]:
        """
        Return normalised weights for `dataset_type`.

        Falls back to universal if the type is unknown.
        Fitted weights take priority over priors.
        """
        key = dataset_type.lower()
        if key in self._fitted:
            return self._normalise(self._fitted[key])
        if key in self._priors:
            return self._normalise(self._priors[key])
        return self._normalise(self._priors["universal"])

    def update(self, dataset_type: str, weights: Dict[str, float]) -> None:
        """Store data-driven fitted weights for a dataset type."""
        self._fitted[dataset_type.lower()] = self._normalise(weights)

    def list_types(self) -> List[str]:
        """Return all known dataset types (priors + fitted)."""
        return sorted(set(self._priors) | set(self._fitted))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"fitted": self._fitted, "priors": self._priors}, f, indent=2)

    def load(self, path: Path) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        self._fitted = data.get("fitted", {})
        self._priors = data.get("priors", self._priors)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total == 0:
            n = len(weights)
            return {k: 1.0 / n for k in weights}
        return {k: v / total for k, v in weights.items()}


# ---------------------------------------------------------------------------
# WeightFitter
# ---------------------------------------------------------------------------

class WeightFitter:
    """
    Fit composite-reward weights from calibration data.

    Two fitting methods are provided:

    Method 1 — "correlation" (default, fast, ~100 samples sufficient)
        w_k = |Pearson_corr(metric_k, y)| / Σ |corr(metric_j, y)|
        y = downstream signal (token_f1 by default)
        Interpretable.  Fast.  Does not account for inter-metric correlation.

    Method 2 — "nnls" (non-negative least squares, recommended N > 200)
        min ||y - X·w||²  s.t. w_k ≥ 0
        X = (N × K) matrix of per-sample metric scores
        y = (N,) downstream signal
        Accounts for inter-metric redundancy.
        Requires scipy.

    Cross-dataset ensemble (fit_cross_dataset)
        Runs either method on each dataset split independently, then pools
        weight vectors via inverse-variance weighting (bootstrapped variance).
        Produces universal weights robust across dataset types.
    """

    N_BOOTSTRAP: int = 200   # bootstrap resamples for variance estimation

    def fit(
        self,
        metric_scores: List[Dict[str, float]],
        downstream_scores: List[float],
        method: str = "correlation",
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Fit weights from per-sample metric scores and a downstream signal.

        Args:
            metric_scores:      List of dicts, one per sample,
                                keys = metric names, values = scores ∈ [0,1].
            downstream_scores:  Scalar quality signal per sample (e.g. token_f1).
            method:             "correlation" | "nnls"
            metric_names:       Subset of metrics to fit.
                                None = use all keys present in metric_scores[0].

        Returns:
            Normalised weight dict summing to 1.
        """
        if not metric_scores:
            raise ValueError("metric_scores must be non-empty")

        keys = metric_names or list(metric_scores[0].keys())
        X = np.array([[s.get(k, 0.0) for k in keys] for s in metric_scores])
        y = np.array(downstream_scores, dtype=float)

        if method == "correlation":
            raw = self._correlation_weights(X, y)
        elif method == "nnls":
            raw = self._nnls_weights(X, y)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'correlation' or 'nnls'.")

        total = raw.sum()
        if total == 0:
            raw = np.ones(len(keys)) / len(keys)
        else:
            raw = raw / total

        return dict(zip(keys, raw.tolist()))

    def fit_cross_dataset(
        self,
        splits: List[Tuple[List[Dict[str, float]], List[float]]],
        method: str = "correlation",
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Fit universal weights by inverse-variance pooling across dataset splits.

        Args:
            splits:  List of (metric_scores, downstream_scores) tuples,
                     one per dataset / split.
            method:  Fitting method to use on each split.
            metric_names: Metric subset to include.

        Returns:
            Normalised universal weight dict.
        """
        if not splits:
            raise ValueError("splits must be non-empty")

        # Fit weights and estimate variance on each split
        all_weights: List[Dict[str, float]] = []
        all_vars:    List[Dict[str, float]] = []

        for metric_scores, downstream_scores in splits:
            w = self.fit(metric_scores, downstream_scores, method=method,
                         metric_names=metric_names)
            v = self._bootstrap_variance(metric_scores, downstream_scores,
                                         method=method, metric_names=metric_names)
            all_weights.append(w)
            all_vars.append(v)

        # Inverse-variance pooling
        keys = list(all_weights[0].keys())
        pooled: Dict[str, float] = {}
        for k in keys:
            weights_k = np.array([w.get(k, 0.0) for w in all_weights])
            vars_k    = np.array([v.get(k, 1e-6) + 1e-6 for v in all_vars])
            inv_vars  = 1.0 / vars_k
            pooled[k] = float(np.sum(weights_k * inv_vars) / np.sum(inv_vars))

        # Normalise
        total = sum(pooled.values())
        return {k: v / total for k, v in pooled.items()}

    # ------------------------------------------------------------------
    # Internal fitting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _correlation_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Absolute Pearson correlation of each column with y."""
        if X.shape[0] < 2:
            return np.ones(X.shape[1])
        corrs = np.array([
            abs(float(np.corrcoef(X[:, j], y)[0, 1]))
            if np.std(X[:, j]) > 0 else 0.0
            for j in range(X.shape[1])
        ])
        return corrs

    @staticmethod
    def _nnls_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Non-negative least squares regression weights."""
        if not _SCIPY_AVAILABLE:
            # Graceful degradation to correlation
            print("Warning: scipy not installed — falling back to correlation weights. "
                  "Install with: pip install scipy")
            return WeightFitter._correlation_weights(X, y)
        w, _ = scipy_nnls(X, y)
        return w

    def _bootstrap_variance(
        self,
        metric_scores: List[Dict[str, float]],
        downstream_scores: List[float],
        method: str = "correlation",
        metric_names: Optional[List[str]] = None,
        n_boot: int = N_BOOTSTRAP,
    ) -> Dict[str, float]:
        """
        Bootstrapped variance of each weight across `n_boot` resamples.
        Used for inverse-variance pooling in fit_cross_dataset.
        """
        n = len(metric_scores)
        keys = metric_names or list(metric_scores[0].keys())
        boot_weights = {k: [] for k in keys}

        rng = np.random.default_rng(seed=42)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_m = [metric_scores[i] for i in idx]
            boot_y = [downstream_scores[i] for i in idx]
            try:
                w = self.fit(boot_m, boot_y, method=method,
                             metric_names=metric_names)
                for k in keys:
                    boot_weights[k].append(w.get(k, 0.0))
            except Exception:
                pass

        return {
            k: float(np.var(boot_weights[k])) if boot_weights[k] else 1e-6
            for k in keys
        }


# ---------------------------------------------------------------------------
# RAGASRewardComputer  (updated — uses HybridMetricBundle + WeightBank)
# ---------------------------------------------------------------------------

class RAGASRewardComputer:
    """
    Computes the composite RAGAS reward using HybridMetricBundle and WeightBank.

    Backward-compatible: still exposes compute_proxy() and compute_full()
    returning List[RAGASReward], same as before.

    Internally delegates metric computation to HybridMetricBundle and
    obtains weights from WeightBank (fitted > prior > universal).
    """

    def __init__(
        self,
        weight_bank: Optional[WeightBank] = None,
        dataset_type: str = "universal",
        llm_model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
    ):
        """
        Args:
            weight_bank:    WeightBank instance.  If None, creates a default
                            one seeded with literature priors.
            dataset_type:   Which weight set to use from the bank.
            llm_model:      LLM for RAGAS NLI-based metrics.
            embedding_model:Embedding model for answer-relevancy.
        """
        self.weight_bank = weight_bank or WeightBank()
        self.dataset_type = dataset_type
        self.metric_bundle = HybridMetricBundle()
        self.llm_model = llm_model
        self.embedding_model = embedding_model

    @property
    def weights(self) -> Dict[str, float]:
        """Return the currently active normalised weights."""
        return self.weight_bank.get_weights(self.dataset_type)

    def _composite(self, scores: Dict[str, float]) -> float:
        """Dot-product of scores with active weights (handles missing keys)."""
        w = self.weights
        return sum(w.get(k, 0.0) * scores.get(k, 0.0) for k in w)

    def compute_proxy(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
    ) -> List[RAGASReward]:
        """Reference-free reward (3 metrics).  Safe at inference time."""
        scores_list = self.metric_bundle.compute_proxy(questions, answers, contexts)
        rewards = []
        for scores in scores_list:
            composite = self._composite(scores)
            rewards.append(RAGASReward(
                faithfulness=scores.get("faithfulness", 0.0),
                answer_relevancy=scores.get("answer_relevancy", 0.0),
                composite=composite,
                mode="proxy",
            ))
        return rewards

    def compute_full(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> List[RAGASReward]:
        """Full reward (up to 9 metrics).  Requires ground truth."""
        scores_list = self.metric_bundle.compute_full(
            questions, answers, contexts, ground_truths
        )
        rewards = []
        for scores in scores_list:
            composite = self._composite(scores)
            rewards.append(RAGASReward(
                faithfulness=scores.get("faithfulness", 0.0),
                answer_relevancy=scores.get("answer_relevancy", 0.0),
                context_precision=scores.get("context_precision", 0.0),
                context_recall=scores.get("context_recall", 0.0),
                composite=composite,
                mode="full",
            ))
        return rewards

    def scalar(self, reward: RAGASReward) -> float:
        return reward.composite


# ---------------------------------------------------------------------------
# ThresholdCalibrator  (updated — fits weights BEFORE threshold sweep)
# ---------------------------------------------------------------------------

class ThresholdCalibrator:
    """
    Calibrates the context-filter threshold τ* on a labelled split.

    New flow (vs. original):
      Step 0  Score all passages via ContextFilter (one-time API cost)
      Step 1  Generate baseline answers using all passages (unfiltered)
      Step 2  Compute full 9-metric bundle on baseline answers
      Step 3  WeightFitter.fit(metric_scores, token_f1) → data-driven weights
      Step 4  Update WeightBank with fitted weights
      Step 5  Sweep τ ∈ {3,4,5,6,7,8} using fitted weights → τ* = argmax R
      Step 6  Persist τ*, fitted weights, WeightBank to JSON

    Critical dependency: weights are fitted in Step 3 BEFORE the sweep (Step 5)
    so that the reward curve is measured with the correct metric importance
    for this dataset.  Using hardcoded or wrong weights shifts the curve peak
    and produces a suboptimal τ*.
    """

    CANDIDATE_THRESHOLDS: List[float] = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    def __init__(
        self,
        context_filter: ContextFilter,
        reward_computer: RAGASRewardComputer,
        generation_fn: Callable[[str, List[str]], str],
        weight_fitter: Optional[WeightFitter] = None,
        weight_method: str = "correlation",
        save_path: Optional[Path] = None,
    ):
        """
        Args:
            context_filter:  Existing ContextFilter (Gemini scorer).
            reward_computer: RAGASRewardComputer with a WeightBank attached.
            generation_fn:   Callable(question, contexts) → answer string.
            weight_fitter:   WeightFitter instance.  None = create default.
            weight_method:   Fitting method passed to WeightFitter.
            save_path:       Where to persist calibration JSON.
        """
        self.context_filter = context_filter
        self.reward_computer = reward_computer
        self.generation_fn = generation_fn
        self.weight_fitter = weight_fitter or WeightFitter()
        self.weight_method = weight_method
        self.save_path = save_path
        self.records: List[CalibrationRecord] = []
        self.optimal_threshold: float = 6.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        questions: List[str],
        passages_list: List[List[str]],
        ground_truths: List[str],
        dataset_type: str = "universal",
        verbose: bool = True,
    ) -> float:
        """
        Run full calibration and return the optimal threshold τ*.

        Args:
            questions:      Calibration questions (100–500 recommended).
            passages_list:  Retrieved passages per question.
            ground_truths:  Gold-standard answers for full metric computation.
            dataset_type:   Dataset identifier stored in WeightBank.
            verbose:        Print progress table.

        Returns:
            Optimal threshold τ* (also stored in self.optimal_threshold).
        """
        bundle = self.reward_computer.metric_bundle

        # ----------------------------------------------------------
        # Step 0 — Score all passages once (reused across all τ)
        # ----------------------------------------------------------
        if verbose:
            print("Step 0 — Scoring passages via ContextFilter (one-time cost)...")
        all_scores: List[List[float]] = []
        all_passages: List[List[str]] = []
        for q, passages in zip(questions, passages_list):
            results = self.context_filter.filter_contexts(q, passages)
            all_scores.append([r.score for r in results])
            all_passages.append([r.passage for r in results])

        # ----------------------------------------------------------
        # Step 1 — Baseline generation with all passages (unfiltered)
        # ----------------------------------------------------------
        if verbose:
            print("Step 1 — Generating baseline answers (all passages, unfiltered)...")
        baseline_answers = [
            self.generation_fn(q, ctx)
            for q, ctx in zip(questions, all_passages)
        ]

        # ----------------------------------------------------------
        # Step 2 — Compute full 9-metric bundle on baseline answers
        # ----------------------------------------------------------
        if verbose:
            print("Step 2 — Computing full metric bundle on baseline answers...")
        full_metrics: List[Dict[str, float]] = bundle.compute_full(
            questions, baseline_answers, all_passages, ground_truths
        )

        # ----------------------------------------------------------
        # Step 3 — Fit weights using token_f1 as downstream signal
        # ----------------------------------------------------------
        downstream_y = [m.get("token_f1", 0.0) for m in full_metrics]
        fitted_weights = self.weight_fitter.fit(
            full_metrics, downstream_y, method=self.weight_method
        )
        if verbose:
            print(f"Step 3 — Fitted weights ({self.weight_method}):")
            for k, v in sorted(fitted_weights.items(), key=lambda x: -x[1]):
                print(f"          {k:<22}: {v:.4f}")

        # ----------------------------------------------------------
        # Step 4 — Update WeightBank
        # ----------------------------------------------------------
        self.reward_computer.weight_bank.update(dataset_type, fitted_weights)
        self.reward_computer.dataset_type = dataset_type

        # ----------------------------------------------------------
        # Step 5 — Threshold sweep using fitted weights
        # ----------------------------------------------------------
        if verbose:
            print("Step 5 — Threshold sweep...")

        records: List[CalibrationRecord] = []
        best_tau, best_reward = self.CANDIDATE_THRESHOLDS[0], -1.0

        for tau in self.CANDIDATE_THRESHOLDS:
            # Filter passages at this threshold
            filtered_contexts = self._apply_threshold(all_scores, all_passages, tau)

            # Generate answers with filtered contexts
            answers = [
                self.generation_fn(q, ctx)
                for q, ctx in zip(questions, filtered_contexts)
            ]

            # Compute full metrics with fitted weights
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
                print(
                    f"  τ={tau:.1f}  composite={mean_composite:.4f}  "
                    f"faith={rec.mean_faithfulness:.4f}  "
                    f"ans_rel={rec.mean_answer_relevancy:.4f}  "
                    f"ctx_prec={rec.mean_context_precision:.4f}  "
                    f"ctx_rec={rec.mean_context_recall:.4f}"
                )

            # Tie-break: prefer higher τ (cheaper inference)
            if mean_composite > best_reward or (
                mean_composite == best_reward and tau > best_tau
            ):
                best_reward = mean_composite
                best_tau = tau

        self.records = records
        self.optimal_threshold = best_tau

        if verbose:
            print(f"\nτ* = {best_tau:.1f}  (mean composite reward = {best_reward:.4f})")

        # ----------------------------------------------------------
        # Step 6 — Persist
        # ----------------------------------------------------------
        if self.save_path:
            self._save(records, best_tau, fitted_weights)

        return best_tau

    def load(self) -> Optional[float]:
        """Load previously saved calibration result.  Returns τ* or None."""
        if self.save_path and Path(self.save_path).exists():
            with open(self.save_path, "r") as f:
                data = json.load(f)
            self.optimal_threshold = data["optimal_threshold"]
            self.records = [
                CalibrationRecord(**{k: v for k, v in r.items()
                                     if k in CalibrationRecord.__dataclass_fields__})
                for r in data["records"]
            ]
            # Restore fitted weights into WeightBank
            if "fitted_weights" in data and "dataset_type" in data:
                self.reward_computer.weight_bank.update(
                    data["dataset_type"], data["fitted_weights"]
                )
                self.reward_computer.dataset_type = data["dataset_type"]
            return self.optimal_threshold
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_threshold(
        all_scores: List[List[float]],
        all_passages: List[List[str]],
        tau: float,
    ) -> List[List[str]]:
        """Apply threshold τ to cached passage scores.  Fallback keeps top-1."""
        result = []
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
        with open(self.save_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Calibration saved → {self.save_path}")


# ---------------------------------------------------------------------------
# RAGASRewardFilter  (public interface unchanged)
# ---------------------------------------------------------------------------

class RAGASRewardFilter:
    """
    H-RRGF — Hybrid RAGAS-Reward-Guided Filtering.

    Wraps the existing LLMFilterPipeline and adds:
    1.  Data-driven weight fitting via WeightFitter + WeightBank.
    2.  Calibrated threshold τ* from ThresholdCalibrator.
    3.  Proxy reward computation after each generation.
    4.  CRAG-inspired corrective retry loop (lower τ if reward < ρ_min).
    5.  Full FilterDiagnostics per sample.

    Public interface is identical to the original RAGASRewardFilter — callers
    do not need to change.

    Usage — inference
    -----------------
    >>> rrgf = RAGASRewardFilter(
    ...     llm_filter=LLMFilterPipeline(...),
    ...     generation_fn=my_qa_fn,
    ...     optimal_threshold=5.0,    # from calibration
    ...     dataset_type="asqa",
    ... )
    >>> answer, filtered_ctx, diag = rrgf.answer(question, passages)

    Usage — calibration (once per dataset split)
    --------------------------------------------
    >>> tau_star = rrgf.calibrate(
    ...     questions, passages_list, ground_truths,
    ...     dataset_type="asqa",
    ...     save_path=Path("outputs/calibration.json"),
    ... )
    """

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
    ):
        """
        Args:
            llm_filter:            Existing LLMFilterPipeline.
            generation_fn:         Callable(question, contexts) → answer.
            reward_computer:       RAGASRewardComputer.  None = build default.
            weight_bank:           WeightBank.  None = build default with priors.
            dataset_type:          Which weight set to use from the bank.
            optimal_threshold:     Starting τ.  Override with calibrate() result.
            min_passages:          Never filter below this many passages.
            max_passages:          Cap passages to this many (top-scored).
            min_reward:            Proxy reward threshold for retry trigger.
            retry_delta:           Threshold step-down per retry.
            max_retries:           Maximum corrective retries.
            use_proxy_reward:      False = skip reward computation (no RAGAS keys).
            weight_fitting_method: "correlation" | "nnls"
        """
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

        # Build / accept reward computer
        if reward_computer is not None:
            self.reward_computer = reward_computer
        else:
            wb = weight_bank or WeightBank()
            self.reward_computer = RAGASRewardComputer(
                weight_bank=wb,
                dataset_type=dataset_type,
            )

        if use_proxy_reward and not _RAGAS_AVAILABLE:
            print(
                "Warning: RAGAS not installed — proxy reward disabled. "
                "Install with: pip install ragas"
            )
            self.use_proxy_reward = False

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        passages: List[str],
    ) -> Tuple[str, List[str], FilterDiagnostics]:
        """
        Run H-RRGF for a single question.

        Steps
        -----
        1. Score passages with ContextFilter (LLM).
        2. Apply calibrated threshold τ to select P*.
        3. Generate answer.
        4. Compute proxy RAGAS reward.
        5. If reward < ρ_min: lower τ by Δτ, retry (CRAG loop).

        Returns
        -------
        answer, filtered_contexts, FilterDiagnostics
        """
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

            answer = self.generation_fn(question, filtered)

            if self.use_proxy_reward:
                proxy_scores = self.reward_computer.metric_bundle.compute_proxy(
                    [question], [answer], [filtered]
                )
                reward = self.reward_computer.compute_proxy(
                    [question], [answer], [filtered]
                )[0]
                full_m = proxy_scores[0]
            else:
                reward = RAGASReward(composite=1.0, mode="skipped")
                full_m = {}

            if retries == 0 or reward.composite > best_reward.composite:
                best_answer = answer
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
        """Run H-RRGF on a batch.  Returns list of (answer, contexts, diag)."""
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
        results = []
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
        """
        Calibrate threshold and fit weights on a labelled split.

        Args:
            questions:      Calibration questions (100–500 recommended).
            passages_list:  Retrieved passages per question.
            ground_truths:  Gold-standard answers.
            dataset_type:   Dataset identifier for WeightBank.  None = use
                            self.dataset_type set at construction.
            save_path:      JSON path to cache calibration result.
            verbose:        Print calibration progress.

        Returns:
            Optimal threshold τ* (also stored in self.optimal_threshold).
        """
        if not _RAGAS_AVAILABLE:
            raise RuntimeError(
                "RAGAS is required for calibration.  "
                "Install with: pip install ragas"
            )

        dt = dataset_type or self.dataset_type

        calibrator = ThresholdCalibrator(
            context_filter=self.llm_filter.context_filter,
            reward_computer=self.reward_computer,
            generation_fn=self.generation_fn,
            weight_fitter=WeightFitter(),
            weight_method=self.weight_fitting_method,
            save_path=save_path,
        )

        # Try loading cached calibration first
        if save_path and Path(save_path).exists():
            cached_tau = calibrator.load()
            if cached_tau is not None:
                if verbose:
                    print(f"Loaded cached calibration: τ* = {cached_tau:.1f}")
                self.optimal_threshold = cached_tau
                return cached_tau

        tau_star = calibrator.calibrate(
            questions, passages_list, ground_truths,
            dataset_type=dt, verbose=verbose
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
            "avg_passages_after":  float(np.mean([d.passages_after for d in diagnostics])),
            "avg_reduction_pct":   float(
                np.mean([
                    (1 - d.passages_after / max(d.passages_before, 1)) * 100
                    for d in diagnostics
                ])
            ),
            "avg_composite_reward":  float(np.mean([d.reward.composite for d in diagnostics])),
            "avg_faithfulness":      float(np.mean([d.reward.faithfulness for d in diagnostics])),
            "avg_answer_relevancy":  float(np.mean([d.reward.answer_relevancy for d in diagnostics])),
            "pct_retried":           float(np.mean([d.retries > 0 for d in diagnostics])) * 100,
            "pct_fallback":          float(np.mean([d.fallback_used for d in diagnostics])) * 100,
            "avg_retries":           float(np.mean([d.retries for d in diagnostics])),
        }
