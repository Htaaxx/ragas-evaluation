#!/usr/bin/env python
"""Evaluate the learned filter with min-FPR threshold selection."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.filtering.config_loader import load_yaml, resolve_path
from src.filtering.data_split import load_and_split
from src.filtering.deberta_filter_evaluator import (
    FilterEvaluator,
    select_threshold_min_fpr,
)
from src.filtering.learned_filter import (
    AnswerQualityClassifier,
    _extract_top1_context,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learned faithfulness filter")
    parser.add_argument(
        "--config",
        default="configs/experiments/filter_training.yaml",
        help="Experiment config YAML",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override model directory",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override results directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    results_dir = resolve_path(args.results_dir or cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    _, val_df, test_df = load_and_split(
        csv_path=str(resolve_path(data_cfg["labeled_csv"])),
        test_ratio=data_cfg["test_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=data_cfg["seed"],
        test_csv_path=str(resolve_path(data_cfg["test_csv"]))
        if data_cfg.get("test_csv")
        else None,
    )

    model_path = str(resolve_path(args.model_path or cfg["model_output"]))
    clf = AnswerQualityClassifier(model_path)
    evaluator = FilterEvaluator()

    val_contexts = [_extract_top1_context(c) for c in val_df["context"].tolist()]
    val_decisions = clf.predict_batch(val_contexts, val_df["answer"].tolist())
    val_confidences = [d.confidence for d in val_decisions]
    val_labels = val_df["label"].tolist()

    min_recall = cfg.get("min_recall_for_threshold", 0.70)
    threshold_result = select_threshold_min_fpr(
        val_confidences, val_labels, min_recall=min_recall,
    )
    best_threshold = threshold_result["threshold"]
    logger.info(
        "Selected threshold=%.3f (FPR=%.4f, recall=%.4f)",
        best_threshold,
        threshold_result["fpr"],
        threshold_result["recall"],
    )

    with open(results_dir / "threshold_selection.json", "w", encoding="utf-8") as fh:
        json.dump(threshold_result, fh, indent=2, default=str)

    test_contexts = [_extract_top1_context(c) for c in test_df["context"].tolist()]
    test_decisions = clf.predict_batch(test_contexts, test_df["answer"].tolist())
    test_preds = [d.confidence >= best_threshold for d in test_decisions]
    test_labels = test_df["label"].tolist()

    learned_result = evaluator.evaluate(test_preds, test_labels)
    learned_result.save(results_dir / "learned_filter_test_results.json")

    baseline_result = evaluator.compute_no_filter_baseline(test_labels)
    comparison = evaluator.compare(
        {"No Filter": baseline_result, "Learned Filter": learned_result},
        save_path=results_dir / "filter_comparison.json",
    )

    print("\n=== COMPARISON TABLE ===")
    for row in comparison:
        print(f"\n{row['strategy']}:")
        for key, value in row.items():
            if key != "strategy":
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    rows = []
    for i, (_, sample) in enumerate(test_df.iterrows()):
        rows.append({
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"][:200],
            "label": int(sample["label"]),
            "predicted": bool(test_preds[i]),
            "confidence": round(test_decisions[i].confidence, 4),
        })
    pd.DataFrame(rows).to_csv(results_dir / "test_predictions.csv", index=False)
    logger.info("Saved results to %s", results_dir)


if __name__ == "__main__":
    main()
