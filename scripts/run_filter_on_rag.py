#!/usr/bin/env python
"""Apply the trained filter to RAG baseline predictions."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from bootstrap import bootstrap

bootstrap()

from rag_filtering.config.loader import load_yaml, resolve_path  # noqa: E402
from rag_filtering.filtering.learned_filter import (  # noqa: E402
    AnswerQualityClassifier,
    _extract_top1_context,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter RAG baseline predictions")
    parser.add_argument(
        "--config",
        default="configs/experiments/asqa_baseline.yaml",
        help="Experiment config YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    model_path = str(resolve_path(cfg["model_path"]))
    predictions_csv = resolve_path(cfg["results"]["predictions_csv"])
    output_csv = resolve_path(cfg["results"]["filtered_csv"])

    clf = AnswerQualityClassifier(model_path)
    df = pd.read_csv(predictions_csv)
    logger.info("Loaded %d RAG predictions from %s", len(df), predictions_csv)

    contexts = [_extract_top1_context(str(c)) for c in df["contexts"].tolist()]
    decisions = clf.predict_batch(contexts, df["predicted_answer"].tolist())

    df["filter_accept"] = [d.accept for d in decisions]
    df["filter_confidence"] = [round(d.confidence, 4) for d in decisions]

    n_accept = sum(d.accept for d in decisions)
    n_reject = len(decisions) - n_accept
    logger.info(
        "Accepted: %d (%.1f%%), Rejected: %d (%.1f%%)",
        n_accept,
        100 * n_accept / len(decisions),
        n_reject,
        100 * n_reject / len(decisions),
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Saved to %s", output_csv)


if __name__ == "__main__":
    main()
