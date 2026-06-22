"""CLI runner for Self-RAG inference over MS MARCO."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
for path in (SRC_DIR, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rag_filtering.config.loader import load_yaml, resolve_path  # noqa: E402
from self_rag_inference.evaluation import compute_answer_metrics  # noqa: E402
from self_rag_inference.msmarco_corpus import (  # noqa: E402
    MSMARCOCorpus,
    MSMARCORetriever,
    load_msmarco_corpus,
)
from self_rag_inference.self_rag_generator import SelfRAGGenerator  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/self_rag_inference.yaml"
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config")
    parser.add_argument("--force-rebuild-index", action="store_true")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Build/load the FAISS index and write retrieval preview without loading Self-RAG.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Override max sample count")
    parser.add_argument("--top-k", type=int, default=None, help="Override retriever top_k")
    return parser.parse_args()


def _results_dir(cfg: Dict[str, Any]) -> Path:
    results_dir = resolve_path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _contexts_json(passages: List[Dict[str, Any]]) -> str:
    return json.dumps(passages, ensure_ascii=False)


def write_retrieval_preview(
    cfg: Dict[str, Any],
    corpus: MSMARCOCorpus,
    retriever: MSMARCORetriever,
    top_k: Optional[int],
) -> Path:
    """Write retrieved contexts without loading the 7B model."""

    rows: List[Dict[str, Any]] = []
    for row in tqdm(corpus.rows, desc="Retrieving"):
        passages = retriever.retrieve(row.question, top_k=top_k)
        rows.append(
            {
                "id": row.row_id,
                "question": row.question,
                "gold_answer": row.gold_answer,
                "contexts": _contexts_json(passages),
                "num_contexts": len(passages),
            }
        )

    out_path = _results_dir(cfg) / "retrieval_preview.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Retrieval preview saved to %s", out_path)
    return out_path


def run_generation(
    cfg: Dict[str, Any],
    corpus: MSMARCOCorpus,
    retriever: MSMARCORetriever,
    top_k: Optional[int],
) -> Dict[str, Any]:
    """Run retrieve-then-generate over the configured corpus."""

    generator = SelfRAGGenerator(cfg)
    generator.load_model()

    output_rows: List[Dict[str, Any]] = []
    predictions: List[str] = []
    references: List[str] = []

    for row in tqdm(corpus.rows, desc="Generating"):
        passages = retriever.retrieve(row.question, top_k=top_k)
        result = generator.generate_answer(row.question, passages)
        best = result.best_candidate

        output_rows.append(
            {
                "id": row.row_id,
                "question": row.question,
                "gold_answer": row.gold_answer,
                "predicted_answer": result.answer,
                "best_context": best.context,
                "best_retrieval_score": best.retrieval_score,
                "best_reflection_score": best.reflection_score,
                "is_relevant": best.is_relevant,
                "is_fully_supported": best.is_fully_supported,
                "utility": best.utility,
                "raw_output": best.raw_output,
                "contexts": _contexts_json(passages),
            }
        )
        predictions.append(result.answer)
        references.append(row.gold_answer)

    metrics = compute_answer_metrics(predictions=predictions, references=references)
    metrics.update(
        {
            "n_samples": len(predictions),
            "config_path": DEFAULT_CONFIG,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "model": cfg["model"]["name"],
            "top_k": top_k or cfg["retriever"]["top_k"],
        }
    )

    results_dir = _results_dir(cfg)
    predictions_path = results_dir / cfg["evaluation"]["predictions_csv"]
    metrics_path = results_dir / cfg["evaluation"]["metrics_json"]

    pd.DataFrame(output_rows).to_csv(predictions_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    logger.info("Predictions saved to %s", predictions_path)
    logger.info("Metrics saved to %s", metrics_path)
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.limit is not None:
        cfg["data"]["max_samples"] = args.limit

    corpus = load_msmarco_corpus(cfg["data"])
    logger.info(
        "Loaded MS MARCO corpus: rows=%d passages=%d",
        len(corpus.rows),
        len(corpus.documents),
    )

    retriever = MSMARCORetriever(cfg=cfg["retriever"], corpus=corpus)
    retriever.build_or_load(force_rebuild=args.force_rebuild_index)

    if args.retrieval_only:
        write_retrieval_preview(cfg, corpus, retriever, top_k=args.top_k)
        return

    metrics = run_generation(cfg, corpus, retriever, top_k=args.top_k)
    logger.info("Final metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
