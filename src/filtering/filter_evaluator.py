"""
Evaluation harness for the answer quality filter.

Computes the six required filtering metrics (Precision, Recall, F1,
Accuracy, Rejection Recall, Rejection Rate) and produces a structured
``FilterResult``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Structured result from a filter evaluation run."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    rejection_precision: float
    rejection_recall: float
    rejection_rate: float
    tp: int
    tn: int
    fp: int
    fn: int

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("Filter results saved to %s", path)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


class FilterEvaluator:
    """Evaluates accept/reject predictions against ground-truth labels.

    Convention
    ----------
    - label = 1 → correct answer (positive class)
    - label = 0 → hallucinated answer (negative class)
    - prediction = True → filter accepts the answer
    - prediction = False → filter rejects the answer
    """

    def evaluate(
        self,
        predictions: Sequence[bool],
        labels: Sequence[int],
    ) -> FilterResult:
        """Compute all six filtering metrics.

        Parameters
        ----------
        predictions:
            Per-sample accept (True) / reject (False) decisions.
        labels:
            Ground-truth labels (1 = correct, 0 = hallucinated).

        Returns
        -------
        FilterResult with all metrics filled in.
        """
        if len(predictions) != len(labels):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs "
                f"{len(labels)} labels"
            )

        tp = sum(p and l == 1 for p, l in zip(predictions, labels))
        tn = sum(not p and l == 0 for p, l in zip(predictions, labels))
        fp = sum(p and l == 0 for p, l in zip(predictions, labels))
        fn = sum(not p and l == 1 for p, l in zip(predictions, labels))

        n = len(labels)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        accuracy = _safe_div(tp + tn, n)
        rejection_precision = _safe_div(tn, tn + fn)
        rejection_recall = _safe_div(tn, tn + fp)
        rejection_rate = _safe_div(tn + fn, n)

        result = FilterResult(
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            rejection_precision=rejection_precision,
            rejection_recall=rejection_recall,
            rejection_rate=rejection_rate,
            tp=tp, tn=tn, fp=fp, fn=fn,
        )

        logger.info(
            "Evaluation (n=%d): P=%.3f R=%.3f F1=%.3f Acc=%.3f RejR=%.3f",
            n, precision, recall, f1, accuracy, rejection_rate,
        )
        return result

    def compute_no_filter_baseline(
        self,
        labels: Sequence[int],
    ) -> FilterResult:
        """Compute the no-filter baseline (accept everything)."""
        predictions = [True] * len(labels)
        return self.evaluate(predictions, labels)

    def compare(
        self,
        results: dict[str, FilterResult],
        save_path: str | Path | None = None,
    ) -> List[dict]:
        """Build a comparison table from named FilterResults.

        Parameters
        ----------
        results:
            Mapping of strategy name to its ``FilterResult``.
        save_path:
            Optional path to persist the comparison as JSON.

        Returns
        -------
        List of dicts (one per strategy), suitable for ``pd.DataFrame``.
        """
        rows = []
        for name, fr in results.items():
            row = {"strategy": name, **fr.to_dict()}
            rows.append(row)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=2)
            logger.info("Comparison table saved to %s", save_path)

        return rows
