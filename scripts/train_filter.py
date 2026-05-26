#!/usr/bin/env python
"""Train the DeBERTa faithfulness filter on labeled_asqa.csv."""

from __future__ import annotations

import argparse
import logging

from bootstrap import bootstrap

bootstrap()

from rag_filtering.config.loader import load_yaml, resolve_path  # noqa: E402
from rag_filtering.filtering.data_split import load_and_split  # noqa: E402
from rag_filtering.filtering.learned_filter import train_classifier  # noqa: E402

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
    )

    logger.info("Starting classifier training …")
    model_path = train_classifier(
        train_df=train_df,
        val_df=val_df,
        output_dir=str(resolve_path(cfg["model_output"])),
    )
    logger.info("Training complete. Model saved to %s", model_path)


if __name__ == "__main__":
    main()
