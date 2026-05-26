#!/usr/bin/env python
"""Ablation studies: training data size and max sequence length."""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from bootstrap import bootstrap

bootstrap()

from rag_filtering.config.loader import load_yaml, resolve_path  # noqa: E402
from rag_filtering.filtering.data_split import load_and_split  # noqa: E402
from rag_filtering.filtering.filter_evaluator import (  # noqa: E402
    FilterEvaluator,
    select_threshold_min_fpr,
)
from rag_filtering.filtering.learned_filter import (  # noqa: E402
    AnswerQualityClassifier,
    _extract_top1_context,
    train_classifier,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run filter ablation studies")
    parser.add_argument(
        "--config",
        default="configs/experiments/ablations.yaml",
        help="Ablation experiment config YAML",
    )
    return parser.parse_args()


def find_best_threshold(
    clf: AnswerQualityClassifier,
    val_df: pd.DataFrame,
    min_recall: float,
) -> float:
    contexts = [_extract_top1_context(c) for c in val_df["context"].tolist()]
    decisions = clf.predict_batch(contexts, val_df["answer"].tolist())
    confidences = [d.confidence for d in decisions]
    result = select_threshold_min_fpr(
        confidences, val_df["label"].tolist(), min_recall=min_recall,
    )
    return float(result["threshold"])


def evaluate_model(
    clf: AnswerQualityClassifier,
    test_df: pd.DataFrame,
    threshold: float,
) -> dict:
    evaluator = FilterEvaluator()
    contexts = [_extract_top1_context(c) for c in test_df["context"].tolist()]
    decisions = clf.predict_batch(contexts, test_df["answer"].tolist())
    preds = [d.confidence >= threshold for d in decisions]
    return evaluator.evaluate(preds, test_df["label"].tolist()).to_dict()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    min_recall = cfg.get("min_recall_for_threshold", 0.70)

    train_df, val_df, test_df = load_and_split(
        csv_path=str(resolve_path(data_cfg["labeled_csv"])),
        test_ratio=data_cfg["test_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=data_cfg["seed"],
    )

    results_dir = resolve_path(cfg["results_dir"])
    models_dir = resolve_path(cfg["models_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    size_rows = []
    for frac in cfg["data_fractions"]:
        tag = f"data_{int(frac * 100)}pct"
        n_samples = int(len(train_df) * frac)
        subset = train_df.sample(n=n_samples, random_state=data_cfg["seed"]).reset_index(
            drop=True,
        )
        out_dir = models_dir / tag
        logger.info("Training on %d/%d samples (%.0f%%) …", n_samples, len(train_df), frac * 100)
        train_classifier(subset, val_df, output_dir=str(out_dir))
        clf = AnswerQualityClassifier(str(out_dir))
        best_t = find_best_threshold(clf, val_df, min_recall)
        metrics = evaluate_model(clf, test_df, best_t)
        size_rows.append({"fraction": frac, "n_train": n_samples, "threshold": best_t, **metrics})

    with open(results_dir / "ablation_data_size.json", "w", encoding="utf-8") as fh:
        json.dump(size_rows, fh, indent=2)

    length_rows = []
    for ml in cfg["max_lengths"]:
        tag = f"maxlen_{ml}"
        out_dir = models_dir / tag
        logger.info("Training with max_length=%d …", ml)
        train_classifier(
            train_df, val_df,
            output_dir=str(out_dir),
            config_overrides={"max_length": ml},
        )
        clf = AnswerQualityClassifier(str(out_dir))
        clf.max_length = ml
        best_t = find_best_threshold(clf, val_df, min_recall)
        metrics = evaluate_model(clf, test_df, best_t)
        length_rows.append({"max_length": ml, "threshold": best_t, **metrics})

    with open(results_dir / "ablation_max_length.json", "w", encoding="utf-8") as fh:
        json.dump(length_rows, fh, indent=2)

    logger.info("All ablations complete. Results in %s", results_dir)


if __name__ == "__main__":
    main()
