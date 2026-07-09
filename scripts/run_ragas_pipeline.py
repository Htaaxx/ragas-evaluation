#!/usr/bin/env python
"""Orchestrator: run the full RAGAS filter pipeline end to end.

Stages:
    1. build_ragas_features  (labeled corpus -> RAGAS features)
    2. train_ragas_filter    (features -> best model + min-FPR threshold)
    3. run_ragas_filter_on_rag (normal-RAG predictions -> accept/reject)
"""

from __future__ import annotations

import argparse
import logging

from bootstrap import bootstrap

bootstrap()

import build_ragas_features  # noqa: E402
import run_ragas_filter_on_rag  # noqa: E402
import train_ragas_filter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/experiments/ragas_filter.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config")
    parser.add_argument(
        "--train-limit", type=int, default=None, help="Cap labeled rows for features"
    )
    parser.add_argument(
        "--apply-limit", type=int, default=None, help="Cap RAG rows to filter"
    )
    parser.add_argument(
        "--skip-features", action="store_true", help="Reuse existing feature CSV"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_features:
        logger.info("=== Stage 1: build RAGAS features ===")
        build_ragas_features.run(args.config, limit=args.train_limit)

    logger.info("=== Stage 2: train RAGAS filter ===")
    train_ragas_filter.run(args.config)

    logger.info("=== Stage 3: apply filter to RAG predictions ===")
    run_ragas_filter_on_rag.run(args.config, limit=args.apply_limit)

    logger.info("RAGAS filter pipeline complete.")


if __name__ == "__main__":
    main()
