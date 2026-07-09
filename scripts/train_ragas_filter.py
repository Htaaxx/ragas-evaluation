#!/usr/bin/env python
"""Stage 2: train the RAGAS-feature filter and select the min-FPR threshold."""

from __future__ import annotations

import argparse
import json
import logging

from bootstrap import bootstrap

bootstrap()

from rag_filtering.config.loader import load_yaml, resolve_path  # noqa: E402
from rag_filtering.filtering.filter_evaluator import (  # noqa: E402
    select_threshold_min_fpr,
)
from rag_filtering.filtering.ragas_filter_trainer import (  # noqa: E402
    RagasFilterTrainer,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/experiments/ragas_filter.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config")
    return parser.parse_args()


def _select_threshold(train_out: dict, min_recall: float, save_path) -> dict:
    """Pick the FPR-min threshold from the best model's test predictions."""
    preds = train_out["test_predictions"]
    best = train_out["best_model_name"]
    best_preds = preds[preds["model"] == best]
    if "y_prob" not in best_preds.columns:
        logger.warning(
            "Best model '%s' has no probabilities; skipping threshold selection.",
            best,
        )
        return {}

    result = select_threshold_min_fpr(
        confidences=best_preds["y_prob"].tolist(),
        labels=best_preds["y_true"].astype(int).tolist(),
        min_recall=min_recall,
    )
    save_path = resolve_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    logger.info(
        "Threshold=%.3f (FPR=%.4f recall=%.4f) saved to %s",
        result["threshold"],
        result["fpr"],
        result["recall"],
        save_path,
    )
    return result


def run(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    tr_cfg = cfg["training"]
    feature_path = resolve_path(cfg["feature_extraction"]["feature_path"])

    trainer = RagasFilterTrainer(
        feature_data=feature_path,
        output_dir=str(resolve_path(tr_cfg["model_output"])),
        label_col=tr_cfg.get("label_col", "label"),
        id_col=tr_cfg.get("id_col", "id"),
        sort_by=tr_cfg.get("sort_by", "f1"),
        test_size=tr_cfg.get("test_size", 0.2),
        random_state=tr_cfg.get("random_state", 42),
    )
    train_out = trainer.run()
    logger.info("Best model: %s -> %s", train_out["best_model_name"], train_out["model_path"])
    logger.info("\n%s", train_out["results_df"].to_string(index=False))

    _select_threshold(
        train_out,
        min_recall=tr_cfg.get("min_recall_for_threshold", 0.70),
        save_path=tr_cfg["threshold_selection_path"],
    )
    return train_out


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
