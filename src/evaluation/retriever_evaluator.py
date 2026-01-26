"""
Retriever evaluation module.

This module provides utilities for evaluating retriever performance
using metrics like Recall@K, MRR, and Precision@K.
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from ..config import RAGConfig


class RetrieverEvaluator:
    """
    Evaluator for retriever models.
    
    Metrics:
    - Recall@K (Top-1, Top-3, Top-5, etc.)
    - Mean Reciprocal Rank (MRR)
    - Precision@K
    """
    
    def __init__(
        self,
        encoder: SentenceTransformer,
        config: RAGConfig,
        device: Optional[str] = None
    ):
        """
        Initialize the retriever evaluator.
        
        Args:
            encoder: SentenceTransformer model to evaluate
            config: Configuration object
            device: Device to use
        """
        self.encoder = encoder
        self.config = config
        self.device = device or config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encoder.to(self.device)
        self.encoder.eval()
    
    def evaluate(
        self,
        df_valid: pd.DataFrame,
        top_k_values: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
        cache_path: Optional[Path] = None,
        rebuild_cache: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate retriever on validation set.
        
        Args:
            df_valid: Validation DataFrame with 'docs' and 'supporting_facts'
            top_k_values: List of K values for Recall@K (default: [1, 3, 5])
            batch_size: Batch size for encoding
            cache_path: Path to cache corpus embeddings
            rebuild_cache: Whether to rebuild cache
            
        Returns:
            Dictionary with evaluation metrics
        """
        top_k_values = top_k_values or self.config.eval_top_k_values
        batch_size = batch_size or self.config.eval_batch_size
        max_k = max(top_k_values)
        
        print(f"\nEvaluating retriever on {len(df_valid)} questions...")
        print(f"   Top-K values: {top_k_values}")
        
        # Build corpus from validation set
        print("Building corpus from validation set...")
        all_titles, all_texts = [], []
        for docs in df_valid["docs"]:
            for doc in docs:
                all_titles.append(doc["title"])
                all_texts.append(doc["text"])
        
        # Deduplicate corpus
        unique_corpus = dict(zip(all_titles, all_texts))
        corpus_titles = list(unique_corpus.keys())
        corpus_texts = list(unique_corpus.values())
        
        print(f"   Corpus size: {len(corpus_texts)} unique documents")
        
        # Encode corpus (with caching)
        if cache_path and not rebuild_cache and Path(cache_path).exists():
            print(f"Loading corpus embeddings from cache: {cache_path}")
            corpus_embs = torch.load(cache_path, map_location=self.device)
        else:
            print(f"Encoding corpus ({len(corpus_texts)} documents)...")
            corpus_embs = self.encoder.encode(
                corpus_texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            
            # Cache embeddings
            if cache_path:
                cache_path = Path(cache_path)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(corpus_embs, cache_path)
                print(f"Cached corpus embeddings to {cache_path}")
        
        # Evaluate on validation set
        recall_at_k = {k: [] for k in top_k_values}
        precision_at_k = {k: [] for k in top_k_values}
        mrrs = []
        
        print(f"\nEvaluating retrieval performance...")
        for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Evaluating"):
            question = row["question"]
            
            # Get gold (relevant) document titles
            try:
                supporting_facts = row["supporting_facts"]
                if isinstance(supporting_facts, str):
                    supporting_facts = ast.literal_eval(supporting_facts)
                
                if isinstance(supporting_facts, dict):
                    gold_titles = set(supporting_facts.get("title", []))
                elif isinstance(supporting_facts, list):
                    gold_titles = set([item[0] for item in supporting_facts])
                else:
                    continue
            except Exception:
                continue
            
            if not gold_titles:
                continue
            
            # Encode question
            q_emb = self.encoder.encode(
                question,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Compute similarities
            similarities = util.cos_sim(q_emb, corpus_embs)[0]
            
            # Get top-K results
            top_indices = torch.topk(similarities, k=max_k).indices.tolist()
            ranked_titles = [corpus_titles[idx] for idx in top_indices]
            
            # Compute metrics for each K
            for k in top_k_values:
                top_k_titles = ranked_titles[:k]
                
                # Recall@K: Is any gold document in top-K?
                has_relevant = any(title in gold_titles for title in top_k_titles)
                recall_at_k[k].append(1.0 if has_relevant else 0.0)
                
                # Precision@K: What fraction of top-K are relevant?
                num_relevant = sum(1 for title in top_k_titles if title in gold_titles)
                precision_at_k[k].append(num_relevant / k)
            
            # MRR: Reciprocal rank of first relevant document
            first_relevant_rank = None
            for rank, title in enumerate(ranked_titles, start=1):
                if title in gold_titles:
                    first_relevant_rank = rank
                    break
            
            if first_relevant_rank:
                mrrs.append(1.0 / first_relevant_rank)
            else:
                mrrs.append(0.0)
        
        # Aggregate metrics
        metrics = {}
        
        for k in top_k_values:
            metrics[f"recall@{k}"] = np.mean(recall_at_k[k])
            metrics[f"precision@{k}"] = np.mean(precision_at_k[k])
        
        metrics["mrr"] = np.mean(mrrs)
        
        # Print results
        print(f"\nRetriever Evaluation Results:")
        print("=" * 50)
        for metric_name, value in metrics.items():
            print(f"   {metric_name:20s}: {value:.4f}")
        print("=" * 50)
        
        return metrics
    
    def evaluate_with_index(
        self,
        df_valid: pd.DataFrame,
        indexer,  # DocumentIndexer
        top_k_values: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Evaluate retriever using an existing FAISS index.
        
        Args:
            df_valid: Validation DataFrame
            indexer: DocumentIndexer with loaded index
            top_k_values: List of K values for metrics
            
        Returns:
            Dictionary with evaluation metrics
        """
        top_k_values = top_k_values or self.config.eval_top_k_values
        max_k = max(top_k_values)
        
        print(f"\nEvaluating retriever with FAISS index...")
        
        # Build title mapping
        title_to_idx = {title: idx for idx, title in enumerate(indexer.doc_titles)}
        
        # Evaluate
        recall_at_k = {k: [] for k in top_k_values}
        mrrs = []
        
        for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Evaluating"):
            question = row["question"]
            
            # Get gold titles
            try:
                supporting_facts = row["supporting_facts"]
                if isinstance(supporting_facts, str):
                    supporting_facts = ast.literal_eval(supporting_facts)
                
                if isinstance(supporting_facts, dict):
                    gold_titles = set(supporting_facts.get("title", []))
                elif isinstance(supporting_facts, list):
                    gold_titles = set([item[0] for item in supporting_facts])
                else:
                    continue
            except Exception:
                continue
            
            # Retrieve documents
            _, _, indices = indexer.search(question, top_k=max_k)
            ranked_titles = [indexer.doc_titles[idx] for idx in indices if idx < len(indexer.doc_titles)]
            
            # Compute metrics
            for k in top_k_values:
                top_k_titles = ranked_titles[:k]
                has_relevant = any(title in gold_titles for title in top_k_titles)
                recall_at_k[k].append(1.0 if has_relevant else 0.0)
            
            # MRR
            first_relevant_rank = None
            for rank, title in enumerate(ranked_titles, start=1):
                if title in gold_titles:
                    first_relevant_rank = rank
                    break
            
            if first_relevant_rank:
                mrrs.append(1.0 / first_relevant_rank)
            else:
                mrrs.append(0.0)
        
        # Aggregate
        metrics = {}
        for k in top_k_values:
            metrics[f"recall@{k}"] = np.mean(recall_at_k[k])
        metrics["mrr"] = np.mean(mrrs)
        
        # Print results
        print(f"\nRetriever Evaluation Results:")
        print("=" * 50)
        for metric_name, value in metrics.items():
            print(f"   {metric_name:20s}: {value:.4f}")
        print("=" * 50)
        
        return metrics
