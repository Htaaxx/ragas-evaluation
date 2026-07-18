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
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

_DATASET_DISPLAY = {
    "asqa": "ASQA",
    "msmarco": "MS MARCO",
    "ms-marco": "MS MARCO",
    "ms_marco": "MS MARCO",
    "wikieval": "WikiEval",
    "wikiEval": "WikiEval",
    "overall": "Overall",
}


def _display_dataset_name(name: str) -> str:
    key = str(name).strip()
    return _DATASET_DISPLAY.get(key, _DATASET_DISPLAY.get(key.lower(), key))


def classification_report_by_dataset(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    pred_col: str = "predicted",
    conf_col: Optional[str] = "confidence",
    dataset_col: str = "dataset",
) -> pd.DataFrame:
    """Build the thesis classification table used by RAGAS / LLM-judge notebooks.

    Columns:
        dataset, num_samples, accepted, acceptance_rate,
        accuracy, precision, recall, f1, roc_auc
    """
    required = {label_col, pred_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for classification report: {sorted(missing)}")

    work = df.copy()
    work[label_col] = work[label_col].astype(int)
    # Accept bool / {0,1} predictions
    work[pred_col] = work[pred_col].astype(int)
    if dataset_col not in work.columns:
        work[dataset_col] = "Overall"

    rows: List[dict] = []

    def _metrics_for(group: pd.DataFrame, dataset_name: str) -> dict:
        y_true = group[label_col]
        y_pred = group[pred_col]
        accepted = int(y_pred.sum())
        n = int(len(group))
        score = None
        if conf_col and conf_col in group.columns:
            score = group[conf_col].astype(float)
        try:
            if y_true.nunique() > 1:
                roc = float(
                    roc_auc_score(y_true, score if score is not None else y_pred)
                )
            else:
                roc = float("nan")
        except ValueError:
            roc = float("nan")
        return {
            "dataset": _display_dataset_name(dataset_name),
            "num_samples": n,
            "accepted": accepted,
            "acceptance_rate": float(accepted / n) if n else 0.0,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": roc,
        }

    if dataset_col in work.columns and work[dataset_col].nunique() >= 1:
        for dataset_name, group in work.groupby(dataset_col, sort=True):
            rows.append(_metrics_for(group, str(dataset_name)))

    rows.append(_metrics_for(work, "Overall"))
    return pd.DataFrame(rows)


def average_classification_reports(reports: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Mean numeric metrics across runs, keyed by ``dataset`` (RAGAS-style)."""
    if not reports:
        return pd.DataFrame(
            columns=[
                "dataset",
                "num_samples",
                "accepted",
                "acceptance_rate",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
        )
    stacked = pd.concat(reports, ignore_index=True)
    numeric = [
        "num_samples",
        "accepted",
        "acceptance_rate",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]
    avg = (
        stacked.groupby("dataset", sort=False)[numeric]
        .mean()
        .reset_index()
    )
    # Keep Overall last; other datasets alphabetical already from concat order
    if "Overall" in set(avg["dataset"]):
        others = avg[avg["dataset"] != "Overall"]
        overall = avg[avg["dataset"] == "Overall"]
        avg = pd.concat([others, overall], ignore_index=True)
    return avg


def _to_python_type(value):
    """Convert NumPy scalar types to native Python types."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


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
        return {
            k: _to_python_type(v)
            for k, v in asdict(self).items()
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

        logger.info("Filter results saved to %s", path)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def select_threshold_min_fpr(
    confidences: Sequence[float],
    labels: Sequence[int],
    min_recall: float = 0.70,
    t_range: tuple = (0.05, 0.99, 0.01),
) -> dict:
    """Select the threshold that minimizes FPR subject to recall >= min_recall.

    This is the thesis-aligned filter decision rule. We want to keep
    hallucinations OUT (low FPR) while still letting through enough
    correct answers (recall floor). F1 is intentionally NOT the
    selection criterion — it can mask catastrophic FPR.

    Parameters
    ----------
    confidences:
        P(faithful) score per sample (e.g., softmax[:, 1] from the
        learned classifier, or entailment probability from NLI).
    labels:
        Ground-truth labels (1 = correct, 0 = hallucinated).
    min_recall:
        Recall floor. We will NOT select a threshold that drops recall
        below this value.
    t_range:
        ``(start, stop_inclusive, step)`` for the threshold sweep.

    Returns
    -------
    Dict with the selected ``threshold``, ``fpr``, ``recall``,
    ``precision``, ``f1``, plus the full sweep table under ``sweep``
    for diagnostics.
    """
    confidences_arr = np.asarray(confidences, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    n_pos = int((labels_arr == 1).sum())
    n_neg = int((labels_arr == 0).sum())

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Threshold sweep needs both classes; got pos={n_pos}, neg={n_neg}"
        )

    start, stop, step = t_range
    thresholds = np.arange(start, stop + step / 2, step)

    sweep = []
    for t in thresholds:
        preds = (confidences_arr >= t).astype(int)
        tp = int(((preds == 1) & (labels_arr == 1)).sum())
        tn = int(((preds == 0) & (labels_arr == 0)).sum())
        fp = int(((preds == 1) & (labels_arr == 0)).sum())
        fn = int(((preds == 0) & (labels_arr == 1)).sum())

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        sweep.append({
            "threshold": float(t),
            "fpr": float(fpr),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })

    # Filter to thresholds that meet the recall floor
    eligible = [s for s in sweep if s["recall"] >= min_recall]
    if not eligible:
        # Fall back to the threshold with highest recall (will not meet floor,
        # but at least gives a usable model). Warn loudly.
        best = max(sweep, key=lambda s: s["recall"])
        logger.warning(
            "No threshold met recall>=%.2f; falling back to highest-recall "
            "threshold t=%.3f (recall=%.3f, fpr=%.3f). "
            "Model is unable to keep recall high; consider retraining.",
            min_recall, best["threshold"], best["recall"], best["fpr"],
        )
    else:
        # Among eligible thresholds, pick the lowest FPR.
        # Ties broken by higher recall (more accepts when both are equally safe).
        best = min(eligible, key=lambda s: (s["fpr"], -s["recall"]))

    logger.info(
        "Threshold selection (min_recall=%.2f): t=%.3f, "
        "FPR=%.3f, recall=%.3f, precision=%.3f, F1=%.3f",
        min_recall, best["threshold"], best["fpr"],
        best["recall"], best["precision"], best["f1"],
    )

    return {
        "threshold": best["threshold"],
        "fpr": best["fpr"],
        "recall": best["recall"],
        "precision": best["precision"],
        "f1": best["f1"],
        "tp": best["tp"], "tn": best["tn"],
        "fp": best["fp"], "fn": best["fn"],
        "min_recall_constraint": float(min_recall),
        "sweep": sweep,
    }


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
        """Compute all six filtering metrics."""
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
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            accuracy=float(accuracy),
            rejection_precision=float(rejection_precision),
            rejection_recall=float(rejection_recall),
            rejection_rate=float(rejection_rate),
            tp=int(tp),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
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
        """Build a comparison table from named FilterResults."""
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