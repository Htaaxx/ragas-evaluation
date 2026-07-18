#!/usr/bin/env python
"""Train the DeBERTa faithfulness filter on labeled_merged.csv."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.filtering.config_loader import load_yaml, resolve_path
from src.filtering.data_split import load_and_split
from src.filtering.learned_filter import train_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeBERTa faithfulness filter")
    parser.add_argument(
        "--config",
        default="configs/experiments/filter_training.yaml",
        help="Experiment config YAML",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override model output directory (e.g. models/answer_filter/run_1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]

    train_df, val_df, _ = load_and_split(
        csv_path=str(resolve_path(data_cfg["labeled_csv"])),
        test_ratio=data_cfg["test_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=data_cfg["seed"],
        test_csv_path=str(resolve_path(data_cfg["test_csv"]))
        if data_cfg.get("test_csv")
        else None,
    )

    output_dir = args.output_dir or cfg["model_output"]
    logger.info("Starting classifier training …")
    model_path = train_classifier(
        train_df=train_df,
        val_df=val_df,
        output_dir=str(resolve_path(output_dir)),
    )
    logger.info("Training complete. Model saved to %s", model_path)


if __name__ == "__main__":
    main()
