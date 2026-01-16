"""
RAGAS evaluation utilities.

This module provides functionality for evaluating RAG systems using the RAGAS framework.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)

from ..models.gemini_llm import create_gemini_llm
from ..config import RAGConfig


class RAGASEvaluator:
    """Evaluator for RAG systems using RAGAS framework."""
    
    # Available metrics
    AVAILABLE_METRICS = {
        'faithfulness': faithfulness,
        'answer_relevancy': answer_relevancy,
        'context_recall': context_recall,
        'context_precision': context_precision,
        'answer_correctness': answer_correctness,
        'answer_similarity': answer_similarity
    }
    
    def __init__(self, embeddings):
        """Initialize the RAGAS evaluator.
        
        Args:
            embeddings: LangChain embeddings instance
        """
        self.embeddings = embeddings
    
    def _create_ragas_llm_and_embeddings(
        self,
        use_openai: bool = False
    ) -> Tuple[Optional[LangchainLLMWrapper], Optional[LangchainEmbeddingsWrapper]]:
        """Create LLM and embeddings wrapped for RAGAS evaluation.
        
        Args:
            use_openai: If True, uses OpenAI. If False, uses Google Gemini.
            
        Returns:
            Tuple of (ragas_llm, ragas_embeddings)
        """
        if use_openai:
            # OpenAI is used by default in RAGAS
            return None, None
        
        # Use Google Gemini
        print("Using Google Gemini for RAGAS evaluation (requires GOOGLE_API_KEY)")
        
        try:
            # Create Gemini LLM with temperature support
            evaluator_llm = create_gemini_llm(temperature=0)
            
        except ValueError as e:
            raise ValueError(str(e))
        except ImportError:
            raise ImportError(
                "langchain-google-genai not installed. Install it with:\n"
                "pip install langchain-google-genai>=2.0.0"
            )
        
        # Wrap for RAGAS
        ragas_llm = LangchainLLMWrapper(evaluator_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        return ragas_llm, ragas_embeddings
    
    def evaluate(
        self,
        ragas_data: Dict[str, List],
        metrics: Optional[List[Any]] = None,
        use_openai: bool = False
    ) -> Dict[str, Any]:
        """Run RAGAS evaluation on prepared data.
        
        Args:
            ragas_data: Dictionary with 'question', 'answer', 'contexts', 'ground_truth' lists
            metrics: List of RAGAS metrics to use. If None, uses all available metrics.
            use_openai: If True, uses OpenAI for evaluation. If False, uses Google Gemini.
            
        Returns:
            Dictionary containing RAGAS evaluation results
        """
        # Get RAGAS LLM and embeddings
        ragas_llm, ragas_embeddings = self._create_ragas_llm_and_embeddings(use_openai)
        
        # Default to all metrics if none specified
        if metrics is None:
            metrics = list(self.AVAILABLE_METRICS.values())
        
        print(f"\nRunning RAGAS evaluation on {len(ragas_data['question'])} samples...")
        print(f"Metrics: {[m.name for m in metrics]}")
        
        # Convert to HuggingFace Dataset format
        dataset = Dataset.from_dict(ragas_data)
        
        # Run RAGAS evaluation with configured LLM
        if use_openai:
            results = evaluate(dataset, metrics=metrics)
        else:
            # Use Google Gemini for evaluation
            results = evaluate(
                dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings
            )
        
        return self._process_results(results, ragas_data, metrics)
    
    def _process_results(
        self,
        results: Any,
        ragas_data: Dict[str, List],
        metrics: List[Any]
    ) -> Dict[str, Any]:
        """Process RAGAS evaluation results into a standardized format.
        
        Args:
            results: Raw RAGAS evaluation results
            ragas_data: Original data used for evaluation
            metrics: List of metrics used
            
        Returns:
            Dictionary with overall_scores and per_sample_scores
        """
        evaluation_results = {
            "overall_scores": {},
            "per_sample_scores": []
        }
        
        # Extract overall scores - RAGAS 0.2.x returns dict-like results
        if hasattr(results, 'items'):
            # RAGAS 0.2.x style
            for key, value in results.items():
                if key not in ['question', 'answer', 'contexts', 'ground_truth']:
                    evaluation_results["overall_scores"][key] = value
        else:
            # RAGAS 0.1.x style
            for metric in metrics:
                metric_name = metric.name
                if hasattr(results, metric_name):
                    evaluation_results["overall_scores"][metric_name] = getattr(results, metric_name)
        
        # Extract per-sample scores if available
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            for idx, row in df.iterrows():
                sample_scores = {
                    "question": ragas_data["question"][idx],
                    "answer": ragas_data["answer"][idx],
                    "ground_truth": ragas_data["ground_truth"][idx],
                }
                # Add metric scores
                for metric in metrics:
                    if metric.name in df.columns:
                        sample_scores[metric.name] = row[metric.name]
                
                evaluation_results["per_sample_scores"].append(sample_scores)
        
        return evaluation_results
    
    @staticmethod
    def get_metrics_by_names(metric_names: List[str]) -> List[Any]:
        """Get metric objects by their names.
        
        Args:
            metric_names: List of metric names
            
        Returns:
            List of metric objects
        """
        metrics = []
        for name in metric_names:
            if name in RAGASEvaluator.AVAILABLE_METRICS:
                metrics.append(RAGASEvaluator.AVAILABLE_METRICS[name])
            else:
                print(f"Warning: Unknown metric '{name}', skipping...")
        return metrics or None
