"""
Retriever evaluation module.

Provides metrics for evaluating retriever performance:
Recall@K, MRR, and Precision@K.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from ..config import RAGConfig
from ..retrieval.indexer import DocumentIndexer

logger = logging.getLogger(__name__)


class RetrieverEvaluator:
    """
    Evaluator for retriever models.

    Metrics: Recall@K, Precision@K, MRR.
    """

    def __init__(
        self,
        encoder: SentenceTransformer,
        config: RAGConfig,
        device: Optional[str] = None,
    ) -> None:
        self.encoder = encoder
        self.config = config
        self.device = (
            device or config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.encoder.to(self.device)
        self.encoder.eval()

    def evaluate(
        self,
        df_valid: pd.DataFrame,
        top_k_values: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
        cache_path: Optional[Path] = None,
        rebuild_cache: bool = False,
    ) -> Dict[str, float]:
        """Evaluate retriever on validation set."""
        top_k_values = top_k_values or self.config.eval_top_k_values
        batch_size = batch_size or self.config.eval_batch_size
        max_k = max(top_k_values)

        logger.info(
            "Evaluating retriever on %d questions (top_k=%s) …",
            len(df_valid), top_k_values,
        )

        corpus_titles, corpus_texts = self._build_validation_corpus(df_valid)
        corpus_embs = self._encode_corpus(
            corpus_texts, batch_size, cache_path, rebuild_cache
        )

        recall_at_k: Dict[int, List[float]] = {k: [] for k in top_k_values}
        precision_at_k: Dict[int, List[float]] = {k: [] for k in top_k_values}
        mrrs: List[float] = []

        for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Evaluating"):
            gold_titles = self._extract_gold_titles(row)
            if gold_titles is None:
                continue

            q_emb = self.encoder.encode(
                row["question"],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            similarities = util.cos_sim(q_emb, corpus_embs)[0]
            top_indices = torch.topk(similarities, k=max_k).indices.tolist()
            ranked_titles = [corpus_titles[idx] for idx in top_indices]

            for k in top_k_values:
                top_k_titles = ranked_titles[:k]
                has_relevant = any(t in gold_titles for t in top_k_titles)
                recall_at_k[k].append(1.0 if has_relevant else 0.0)
                num_relevant = sum(1 for t in top_k_titles if t in gold_titles)
                precision_at_k[k].append(num_relevant / k)

            mrrs.append(self._mrr(ranked_titles, gold_titles))

        metrics: Dict[str, float] = {}
        for k in top_k_values:
            metrics[f"recall@{k}"] = float(np.mean(recall_at_k[k]))
            metrics[f"precision@{k}"] = float(np.mean(precision_at_k[k]))
        metrics["mrr"] = float(np.mean(mrrs))

        logger.info("Retriever Evaluation Results:")
        logger.info("=" * 50)
        for name, value in metrics.items():
            logger.info("   %-20s: %.4f", name, value)
        logger.info("=" * 50)

        return metrics

    def evaluate_with_index(
        self,
        df_valid: pd.DataFrame,
        indexer: DocumentIndexer,
        top_k_values: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Evaluate retriever using an existing FAISS index."""
        top_k_values = top_k_values or self.config.eval_top_k_values
        max_k = max(top_k_values)

        logger.info("Evaluating retriever with FAISS index …")

        recall_at_k: Dict[int, List[float]] = {k: [] for k in top_k_values}
        mrrs: List[float] = []

        for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Evaluating"):
            gold_titles = self._extract_gold_titles(row)
            if gold_titles is None:
                continue

            _, _, indices = indexer.search(row["question"], top_k=max_k)
            ranked_titles = [
                indexer.doc_titles[idx]
                for idx in indices
                if idx < len(indexer.doc_titles)
            ]

            for k in top_k_values:
                top_k_titles = ranked_titles[:k]
                has_relevant = any(t in gold_titles for t in top_k_titles)
                recall_at_k[k].append(1.0 if has_relevant else 0.0)

            mrrs.append(self._mrr(ranked_titles, gold_titles))

        metrics: Dict[str, float] = {}
        for k in top_k_values:
            metrics[f"recall@{k}"] = float(np.mean(recall_at_k[k]))
        metrics["mrr"] = float(np.mean(mrrs))

        logger.info("Retriever Evaluation Results:")
        for name, value in metrics.items():
            logger.info("   %-20s: %.4f", name, value)

        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_validation_corpus(
        df_valid: pd.DataFrame,
    ) -> tuple[list[str], list[str]]:
        all_titles: List[str] = []
        all_texts: List[str] = []
        for docs in df_valid["docs"]:
            for doc in docs:
                all_titles.append(doc["title"])
                all_texts.append(doc["text"])

        unique = dict(zip(all_titles, all_texts))
        logger.info("Corpus size: %d unique documents", len(unique))
        return list(unique.keys()), list(unique.values())

    def _encode_corpus(
        self,
        corpus_texts: List[str],
        batch_size: int,
        cache_path: Optional[Path],
        rebuild_cache: bool,
    ) -> torch.Tensor:
        if cache_path and not rebuild_cache and Path(cache_path).exists():
            logger.info("Loading corpus embeddings from cache: %s", cache_path)
            return torch.load(cache_path, map_location=self.device)

        logger.info("Encoding corpus (%d documents) …", len(corpus_texts))
        corpus_embs = self.encoder.encode(
            corpus_texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        if cache_path:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(corpus_embs, cache_path)
            logger.info("Cached corpus embeddings to %s", cache_path)

        return corpus_embs

    @staticmethod
    def _extract_gold_titles(row: pd.Series) -> Optional[Set[str]]:
        try:
            sf = row["supporting_facts"]
            if isinstance(sf, str):
                sf = ast.literal_eval(sf)
            if isinstance(sf, dict):
                titles = set(sf.get("title", []))
            elif isinstance(sf, list):
                titles = {item[0] for item in sf}
            else:
                return None
            return titles if titles else None
        except Exception:
            return None

    @staticmethod
    def _mrr(ranked_titles: List[str], gold_titles: Set[str]) -> float:
        for rank, title in enumerate(ranked_titles, start=1):
            if title in gold_titles:
                return 1.0 / rank
        return 0.0
