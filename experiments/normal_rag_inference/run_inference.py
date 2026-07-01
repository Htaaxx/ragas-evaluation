"""CLI runner for normal RAG inference over configured QA datasets."""

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

from normal_rag_inference.dataset import (  # noqa: E402
    NormalRAGRetriever,
    QACorpus,
    load_qa_corpus,
)
from normal_rag_inference.evaluation import (  # noqa: E402
    compute_answer_metrics,
    compute_grouped_metrics,
    compute_output_diagnostics,
)
from normal_rag_inference.generator import NormalRAGGenerator  # noqa: E402
from rag_filtering.config.loader import load_yaml, resolve_path  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/normal_rag_merged.yaml"
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config")
    parser.add_argument("--force-rebuild-index", action="store_true")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Build/load FAISS and write retrieval preview without loading the generator.",
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
    corpus: QACorpus,
    retriever: NormalRAGRetriever,
    top_k: Optional[int],
) -> Path:
    """Write retrieved contexts without loading the generator."""

    rows: List[Dict[str, Any]] = []
    for row in tqdm(corpus.rows, desc="Retrieving"):
        passages = retriever.retrieve(row.question, top_k=top_k)
        output_row = {
            "id": row.row_id,
            "question": row.question,
            "gold_answer": row.reference_answer,
            "contexts": _contexts_json(passages),
            "num_contexts": len(passages),
        }
        output_row.update(row.metadata)
        rows.append(output_row)

    out_path = _results_dir(cfg) / "retrieval_preview.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Retrieval preview saved to %s", out_path)
    return out_path


def run_generation(
    cfg: Dict[str, Any],
    config_path: str,
    corpus: QACorpus,
    retriever: NormalRAGRetriever,
    top_k: Optional[int],
) -> Dict[str, Any]:
    """Run retrieve-then-generate over the configured corpus."""

    generator = NormalRAGGenerator(cfg)
    generator.load_model()

    output_rows: List[Dict[str, Any]] = []
    predictions: List[str] = []
    references: List[str] = []

    for row in tqdm(corpus.rows, desc="Generating"):
        passages = retriever.retrieve(row.question, top_k=top_k)
        result = generator.generate_answer(row.question, passages)
        best = result.best_candidate

        output_row = {
            "id": row.row_id,
            "question": row.question,
            "gold_answer": row.reference_answer,
            "predicted_answer": result.answer,
            "best_context": best.context,
            "best_retrieval_score": best.retrieval_score,
            "raw_output": best.raw_output,
            "contexts": _contexts_json(passages),
        }
        output_row.update(row.metadata)
        output_rows.append(output_row)
        predictions.append(result.answer)
        references.append(row.reference_answer)

    metrics = compute_answer_metrics(predictions=predictions, references=references)
    metrics.update(compute_output_diagnostics(predictions))
    metrics.update(
        {
            "n_samples": len(predictions),
            "config_path": config_path,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "model": cfg["model"]["name"],
            "top_k": top_k or cfg["retriever"]["top_k"],
            "dataset_type": corpus.dataset_type,
        }
    )

    group_fields = list(cfg["evaluation"].get("group_fields", []))
    if group_fields:
        metrics["grouped"] = compute_grouped_metrics(output_rows, group_fields=group_fields)

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

    corpus = load_qa_corpus(cfg["data"])
    logger.info(
        "Loaded QA corpus: dataset_type=%s rows=%d passages=%d",
        corpus.dataset_type,
        len(corpus.rows),
        len(corpus.documents),
    )

    retriever = NormalRAGRetriever(cfg=cfg["retriever"], corpus=corpus)
    retriever.build_or_load(force_rebuild=args.force_rebuild_index)

    if args.retrieval_only:
        write_retrieval_preview(cfg, corpus, retriever, top_k=args.top_k)
        return

    metrics = run_generation(
        cfg=cfg,
        config_path=args.config,
        corpus=corpus,
        retriever=retriever,
        top_k=args.top_k,
    )
    logger.info("Final metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
