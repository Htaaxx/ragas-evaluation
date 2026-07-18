"""Filter metric and threshold selection tests."""

from __future__ import annotations

import pandas as pd

from src.filtering.deberta_filter_evaluator import (
    FilterEvaluator,
    classification_report_by_dataset,
    select_threshold_min_fpr,
)


def test_filter_result_schema() -> None:
    evaluator = FilterEvaluator()
    result = evaluator.evaluate([True, False, True, False], [1, 0, 1, 0])
    d = result.to_dict()
    for key in ("precision", "recall", "f1", "accuracy", "tp", "tn", "fp", "fn"):
        assert key in d


def test_select_threshold_min_fpr_prefers_low_fpr() -> None:
    confidences = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    picked = select_threshold_min_fpr(confidences, labels, min_recall=0.5)
    assert picked["threshold"] >= 0.05
    assert picked["recall"] >= 0.5
    assert "fpr" in picked


def test_classification_report_by_dataset_schema() -> None:
    df = pd.DataFrame(
        {
            "label": [1, 1, 0, 0, 1, 0],
            "predicted": [1, 0, 0, 1, 1, 0],
            "confidence": [0.9, 0.4, 0.2, 0.7, 0.8, 0.1],
            "dataset": ["asqa", "asqa", "asqa", "asqa", "msmarco", "msmarco"],
        }
    )
    report = classification_report_by_dataset(df)
    assert list(report.columns) == [
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
    assert "Overall" in set(report["dataset"])
    assert "ASQA" in set(report["dataset"])
    assert "MS MARCO" in set(report["dataset"])
