#!/usr/bin/env python
"""Sanity check: verify lexical metrics separate correct vs hallucinated answers."""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from bootstrap import bootstrap

bootstrap()

from rag_filtering.config.loader import load_yaml, resolve_path  # noqa: E402
from rag_filtering.filtering.data_split import load_and_split  # noqa: E402
from rag_filtering.filtering.metrics import AnswerMetricBundle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metric separation sanity check")
    parser.add_argument(
        "--config",
        default="configs/experiments/filter_training.yaml",
        help="Experiment config YAML (uses data paths)",
    )
    parser.add_argument("--n-samples", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    results_dir = resolve_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    train_df, _, _ = load_and_split(
        csv_path=str(resolve_path(data_cfg["labeled_csv"])),
        test_ratio=data_cfg["test_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=data_cfg["seed"],
    )

    base_ids = train_df["id"].str.replace(r"b$", "", regex=True).unique()[: args.n_samples]
    correct_f1, correct_rl = [], []
    hallu_f1, hallu_rl = [], []
    bundle = AnswerMetricBundle()

    for bid in base_ids:
        rows = train_df[train_df["id"].str.replace(r"b$", "", regex=True) == bid]
        correct_row = rows[rows["label"] == 1].iloc[0]
        hallu_row = rows[rows["label"] == 0].iloc[0]
        ground_truth = correct_row["answer"]

        correct_f1.append(bundle.token_f1(correct_row["answer"], ground_truth))
        correct_rl.append(bundle.rouge_l(correct_row["answer"], ground_truth))
        hallu_f1.append(bundle.token_f1(hallu_row["answer"], ground_truth))
        hallu_rl.append(bundle.rouge_l(hallu_row["answer"], ground_truth))

    results = {
        "correct_token_f1": {"mean": float(np.mean(correct_f1)), "std": float(np.std(correct_f1))},
        "correct_rouge_l": {"mean": float(np.mean(correct_rl)), "std": float(np.std(correct_rl))},
        "hallucinated_token_f1": {"mean": float(np.mean(hallu_f1)), "std": float(np.std(hallu_f1))},
        "hallucinated_rouge_l": {"mean": float(np.mean(hallu_rl)), "std": float(np.std(hallu_rl))},
        "gap_token_f1": float(np.mean(correct_f1) - np.mean(hallu_f1)),
        "gap_rouge_l": float(np.mean(correct_rl) - np.mean(hallu_rl)),
        "n_samples": args.n_samples,
    }

    print("\n=== METRIC SEPARATION SANITY CHECK ===")
    print(f"Gap (token_f1): {results['gap_token_f1']:.4f}")
    print(f"Gap (rouge_l):  {results['gap_rouge_l']:.4f}")

    out_path = results_dir / "metric_separation_sanity_check.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
