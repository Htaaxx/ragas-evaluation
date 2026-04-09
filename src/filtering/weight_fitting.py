"""
Weight fitting and banking for composite-reward computation.

WeightBank — stores per-dataset-type weight vectors (fitted > prior > universal).
WeightFitter — fits weights from calibration data via correlation or NNLS.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_models import LITERATURE_PRIORS

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import nnls as scipy_nnls

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# WeightBank
# ---------------------------------------------------------------------------


class WeightBank:
    """
    Stores composite-reward weights per dataset type.

    Priority: fitted weights > literature prior > universal prior.
    Supports JSON round-trip for persistence.
    """

    def __init__(self) -> None:
        self._priors: Dict[str, Dict[str, float]] = {
            k: dict(v) for k, v in LITERATURE_PRIORS.items()
        }
        self._fitted: Dict[str, Dict[str, float]] = {}

    def get_weights(self, dataset_type: str = "universal") -> Dict[str, float]:
        """Return normalised weights for *dataset_type*."""
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
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"fitted": self._fitted, "priors": self._priors}, fh, indent=2)
        logger.info("WeightBank saved -> %s", path)

    def load(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._fitted = data.get("fitted", {})
        self._priors = data.get("priors", self._priors)
        logger.info("WeightBank loaded <- %s", path)

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

    Method 1 — "correlation" (default, fast, ~100 samples sufficient)
    Method 2 — "nnls" (non-negative least squares, recommended N > 200)
    """

    N_BOOTSTRAP: int = 200

    def fit(
        self,
        metric_scores: List[Dict[str, float]],
        downstream_scores: List[float],
        method: str = "correlation",
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Fit weights from per-sample metric scores and a downstream signal."""
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
            raise ValueError(f"Unknown method '{method}'.")

        total = raw.sum()
        raw = raw / total if total > 0 else np.ones(len(keys)) / len(keys)
        return dict(zip(keys, raw.tolist()))

    def fit_cross_dataset(
        self,
        splits: List[Tuple[List[Dict[str, float]], List[float]]],
        method: str = "correlation",
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Fit universal weights by inverse-variance pooling across splits."""
        if not splits:
            raise ValueError("splits must be non-empty")

        all_weights: List[Dict[str, float]] = []
        all_vars: List[Dict[str, float]] = []

        for metric_scores, downstream_scores in splits:
            w = self.fit(metric_scores, downstream_scores, method=method,
                         metric_names=metric_names)
            v = self._bootstrap_variance(metric_scores, downstream_scores,
                                         method=method, metric_names=metric_names)
            all_weights.append(w)
            all_vars.append(v)

        keys = list(all_weights[0].keys())
        pooled: Dict[str, float] = {}
        for k in keys:
            weights_k = np.array([w.get(k, 0.0) for w in all_weights])
            vars_k = np.array([v.get(k, 1e-6) + 1e-6 for v in all_vars])
            inv_vars = 1.0 / vars_k
            pooled[k] = float(np.sum(weights_k * inv_vars) / np.sum(inv_vars))

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
        return np.array([
            abs(float(np.corrcoef(X[:, j], y)[0, 1]))
            if np.std(X[:, j]) > 0 else 0.0
            for j in range(X.shape[1])
        ])

    @staticmethod
    def _nnls_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Non-negative least squares regression weights."""
        if not _SCIPY_AVAILABLE:
            logger.warning(
                "scipy not installed — falling back to correlation weights."
            )
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
        """Bootstrapped variance of each weight across resamples."""
        n = len(metric_scores)
        keys = metric_names or list(metric_scores[0].keys())
        boot_weights: Dict[str, List[float]] = {k: [] for k in keys}

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
