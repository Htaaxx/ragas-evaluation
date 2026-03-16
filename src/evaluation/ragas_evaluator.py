"""
RAGAS evaluation metrics for RAG systems.

This module implements comprehensive evaluation using the RAGAS framework:
- Faithfulness: Answer grounded in context
- Answer Relevancy: Answer addresses the question
- Context Precision: Relevant contexts ranked higher
- Context Recall: All relevant contexts retrieved
"""

from typing import List, Dict, Optional
import pandas as pd
from tqdm.auto import tqdm

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_relevancy,
        answer_correctness,
        answer_similarity
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Warning: RAGAS not installed. Install with: pip install ragas")


class RAGASEvaluator:
    """
    Evaluator using RAGAS metrics for comprehensive RAG evaluation.
    """
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        llm_model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002"
    ):
        """
        Initialize RAGAS evaluator.
        
        Args:
            metrics: List of metrics to compute. If None, uses all available metrics.
            llm_model: LLM model for LLM-based metrics
            embedding_model: Embedding model for embedding-based metrics
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS not installed. Install with: pip install ragas")
        
        # Default metrics
        self.available_metrics = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'context_recall': context_recall,
            'context_relevancy': context_relevancy,
            'answer_correctness': answer_correctness,
            'answer_similarity': answer_similarity
        }
        
        if metrics is None:
            # Use core metrics by default
            self.metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        else:
            self.metrics = [self.available_metrics[m] for m in metrics if m in self.available_metrics]
        
        self.llm_model = llm_model
        self.embedding_model = embedding_model
    
    def prepare_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dataset:
        """
        Prepare dataset in RAGAS format.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context lists (one per question)
            ground_truths: Optional list of ground truth answers
            
        Returns:
            HuggingFace Dataset
        """
        data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts
        }
        
        if ground_truths is not None:
            data['ground_truth'] = ground_truths
        
        return Dataset.from_dict(data)
    
    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate RAG system using RAGAS metrics.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context lists
            ground_truths: Optional ground truth answers
            show_progress: Show progress bar
            
        Returns:
            Dictionary of metric scores
        """
        # Prepare dataset
        dataset = self.prepare_dataset(questions, answers, contexts, ground_truths)
        
        # Run evaluation
        if show_progress:
            print(f"Evaluating {len(questions)} samples with RAGAS metrics...")
        
        result = evaluate(
            dataset,
            metrics=self.metrics
        )
        
        return result
    
    def evaluate_from_dataframe(
        self,
        df: pd.DataFrame,
        question_col: str = 'question',
        answer_col: str = 'predicted_answer',
        contexts_col: str = 'contexts',
        ground_truth_col: Optional[str] = 'gold_answer',
        show_progress: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate from a pandas DataFrame.
        
        Args:
            df: DataFrame with predictions
            question_col: Column name for questions
            answer_col: Column name for answers
            contexts_col: Column name for contexts
            ground_truth_col: Column name for ground truth (optional)
            show_progress: Show progress bar
            
        Returns:
            Dictionary of metric scores
        """
        questions = df[question_col].tolist()
        answers = df[answer_col].tolist()
        
        # Handle contexts (may be string or list)
        contexts = []
        for ctx in df[contexts_col]:
            if isinstance(ctx, str):
                # Try to parse as list
                try:
                    import ast
                    ctx = ast.literal_eval(ctx)
                except:
                    ctx = [ctx]
            elif not isinstance(ctx, list):
                ctx = [str(ctx)]
            contexts.append(ctx)
        
        ground_truths = None
        if ground_truth_col and ground_truth_col in df.columns:
            ground_truths = df[ground_truth_col].tolist()
        
        return self.evaluate(questions, answers, contexts, ground_truths, show_progress)
    
    def compare_systems(
        self,
        system1_df: pd.DataFrame,
        system2_df: pd.DataFrame,
        system1_name: str = "System 1",
        system2_name: str = "System 2",
        **kwargs
    ) -> pd.DataFrame:
        """
        Compare two RAG systems side by side.
        
        Args:
            system1_df: DataFrame for first system
            system2_df: DataFrame for second system
            system1_name: Name of first system
            system2_name: Name of second system
            **kwargs: Additional arguments for evaluate_from_dataframe
            
        Returns:
            Comparison DataFrame
        """
        print(f"Evaluating {system1_name}...")
        results1 = self.evaluate_from_dataframe(system1_df, **kwargs)
        
        print(f"\nEvaluating {system2_name}...")
        results2 = self.evaluate_from_dataframe(system2_df, **kwargs)
        
        # Create comparison table
        comparison = []
        for metric in results1.keys():
            if metric in results2:
                delta = results2[metric] - results1[metric]
                delta_pct = (delta / results1[metric] * 100) if results1[metric] != 0 else 0
                
                comparison.append({
                    'Metric': metric,
                    system1_name: f"{results1[metric]:.4f}",
                    system2_name: f"{results2[metric]:.4f}",
                    'Δ': f"{delta:+.4f}",
                    'Δ%': f"{delta_pct:+.1f}%"
                })
        
        return pd.DataFrame(comparison)


def evaluate_rag_pipeline(
    predictions_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    question_col: str = 'question',
    answer_col: str = 'predicted_answer',
    contexts_col: str = 'contexts',
    ground_truth_col: Optional[str] = 'gold_answer'
) -> Dict[str, float]:
    """
    Convenience function to evaluate a RAG pipeline from predictions DataFrame.
    
    Args:
        predictions_df: DataFrame with predictions
        metrics: List of metrics to compute
        question_col: Column name for questions
        answer_col: Column name for answers
        contexts_col: Column name for contexts
        ground_truth_col: Column name for ground truth
        
    Returns:
        Dictionary of metric scores
    """
    evaluator = RAGASEvaluator(metrics=metrics)
    return evaluator.evaluate_from_dataframe(
        predictions_df,
        question_col=question_col,
        answer_col=answer_col,
        contexts_col=contexts_col,
        ground_truth_col=ground_truth_col
    )


def compare_rag_systems(
    system1_df: pd.DataFrame,
    system2_df: pd.DataFrame,
    system1_name: str = "Normal RAG",
    system2_name: str = "Filtered RAG",
    metrics: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to compare two RAG systems.
    
    Args:
        system1_df: DataFrame for first system
        system2_df: DataFrame for second system
        system1_name: Name of first system
        system2_name: Name of second system
        metrics: List of metrics to compute
        **kwargs: Additional arguments for evaluate_from_dataframe
        
    Returns:
        Comparison DataFrame
    """
    evaluator = RAGASEvaluator(metrics=metrics)
    return evaluator.compare_systems(
        system1_df,
        system2_df,
        system1_name=system1_name,
        system2_name=system2_name,
        **kwargs
    )
