#!/usr/bin/env python
"""Stage 1: build RAGAS features from the labeled corpus (OpenAI backend)."""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import pandas as pd

from bootstrap import bootstrap

bootstrap()

from dotenv import load_dotenv  # noqa: E402

from rag_filtering.config.loader import load_yaml, resolve_path  # noqa: E402
from rag_filtering.evaluation.ragas_wrapper import RAGAS  # noqa: E402
from rag_filtering.filtering.ragas_feature_extractor import (  # noqa: E402
    RagasFeatureExtractor,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/experiments/ragas_filter.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override feature_extraction.max_samples (cost control)",
    )
    return parser.parse_args()


def run(config_path: str, limit: Optional[int] = None) -> str:
    load_dotenv()
    cfg = load_yaml(config_path)
    ragas_cfg = cfg["ragas"]
    fe_cfg = cfg["feature_extraction"]

    max_samples = limit if limit is not None else fe_cfg.get("max_samples")
    labeled_csv = resolve_path(fe_cfg["labeled_csv"])
    feature_path = resolve_path(fe_cfg["feature_path"])
    checkpoint_path = resolve_path(fe_cfg["checkpoint_path"])

    df = pd.read_csv(labeled_csv)
    if max_samples:
        df = df.head(int(max_samples)).copy()
    logger.info("Loaded %d labeled rows from %s", len(df), labeled_csv)

    ragas = RAGAS(
        metrics=ragas_cfg["metrics"],
        llm_model=ragas_cfg["llm_model"],
        embedding_model=ragas_cfg["embedding_model"],
        temperature=ragas_cfg.get("temperature", 0.0),
    )
    extractor = RagasFeatureExtractor(
        ragas_evaluator=ragas,
        feature_cols=ragas_cfg["metrics"],
    )
    extractor.transform(
        data=df,
        feature_path=feature_path,
        checkpoint_path=checkpoint_path,
        batch_size=int(ragas_cfg.get("batch_size", 20)),
    )
    logger.info("RAGAS features saved to %s", feature_path)
    return str(feature_path)


def main() -> None:
    args = parse_args()
    run(args.config, args.limit)


if __name__ == "__main__":
    main()
