"""Filter metric and threshold selection tests."""

from __future__ import annotations

from rag_filtering.filtering.filter_evaluator import (
    FilterEvaluator,
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
