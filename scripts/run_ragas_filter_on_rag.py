#!/usr/bin/env python
"""Stage 3: apply the trained RAGAS filter to normal-RAG predictions.

This is inference-only: the corpus ``label`` describes the original corpus
answer, not the generated ``predicted_answer``, so it is dropped before
filtering and no accept/reject ground-truth metric is computed here.
"""

from __future__ import annotations

import argparse
import json
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
from rag_filtering.filtering.ragas_filter import RagasFilter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/experiments/ragas_filter.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config")
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap RAG rows to filter (cost control)"
    )
    return parser.parse_args()


def _resolve_model_path(model_output) -> str:
    model_dir = resolve_path(model_output)
    meta = model_dir / "training_metadata.json"
    if meta.exists():
        with open(meta, "r", encoding="utf-8") as fh:
            best = json.load(fh).get("best_model_name")
        candidate = model_dir / f"{best}.joblib"
        if candidate.exists():
            return str(candidate)
    joblibs = sorted(model_dir.glob("*.joblib"))
    if not joblibs:
        raise FileNotFoundError(f"No .joblib model found in {model_dir}")
    return str(joblibs[0])


def _resolve_threshold(path) -> float:
    path = resolve_path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return float(json.load(fh)["threshold"])
    logger.warning("No threshold_selection.json at %s; falling back to 0.5", path)
    return 0.5


def run(config_path: str, limit: Optional[int] = None) -> str:
    load_dotenv()
    cfg = load_yaml(config_path)
    ragas_cfg = cfg["ragas"]
    tr_cfg = cfg["training"]
    apply_cfg = cfg["apply"]

    model_path = _resolve_model_path(tr_cfg["model_output"])
    threshold = _resolve_threshold(tr_cfg["threshold_selection_path"])

    predictions_csv = resolve_path(apply_cfg["rag_predictions_csv"])
    df = pd.read_csv(predictions_csv)
    if limit:
        df = df.head(int(limit)).copy()
    # Corpus label does not describe the generated answer -> drop for inference.
    df = df.drop(columns=["label"], errors="ignore")
    logger.info("Loaded %d RAG predictions from %s", len(df), predictions_csv)

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
    ragas_filter = RagasFilter(
        model_path=model_path,
        feature_extractor=extractor,
        output_dir=str(resolve_path(apply_cfg["results_dir"])),
        threshold=threshold,
    )

    out = ragas_filter.predict(
        data=df,
        feature_path=resolve_path(apply_cfg["feature_path"]),
        checkpoint_path=resolve_path(apply_cfg["checkpoint_path"]),
        filter_path=resolve_path(apply_cfg["filtered_csv"]),
        batch_size=int(ragas_cfg.get("batch_size", 20)),
    )

    n_accept = int((out["filter_label"] == 1).sum())
    n_reject = len(out) - n_accept
    logger.info(
        "Accepted: %d (%.1f%%), Rejected: %d (%.1f%%) [threshold=%.3f]",
        n_accept,
        100 * n_accept / max(len(out), 1),
        n_reject,
        100 * n_reject / max(len(out), 1),
        threshold,
    )
    logger.info("Filtered predictions saved to %s", resolve_path(apply_cfg["filtered_csv"]))
    return str(resolve_path(apply_cfg["filtered_csv"]))


def main() -> None:
    args = parse_args()
    run(args.config, args.limit)


if __name__ == "__main__":
    main()
