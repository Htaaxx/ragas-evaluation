"""
RAGAS evaluator for RAG systems.

Supports:
- configurable OpenAI-compatible LLM + embedding models
- configurable API key from notebook / script
- dataframe evaluation
- system comparison
- black-box RAG evaluation metrics

Example:
    evaluator = RAGASEvaluator(
        metrics=[
            "faithfulness",
            "answer_relevancy",
        ],
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    results = evaluator.evaluate_from_dataframe(df)
"""

from __future__ import annotations

import ast
import logging
import os
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# IMPORTS
# ============================================================

try:
    from datasets import Dataset

    from ragas import evaluate

    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        answer_similarity,
        context_precision,
        context_recall,
        context_relevancy,
        faithfulness,
    )

    from langchain_openai import (
        ChatOpenAI,
        OpenAIEmbeddings,
    )

    RAGAS_AVAILABLE = True

except ImportError:

    RAGAS_AVAILABLE = False

    logger.warning(
        "RAGAS dependencies missing.\n"
        "Install:\n"
        "pip install ragas langchain-openai datasets"
    )


# ============================================================
# MAIN CLASS
# ============================================================

class RAGASEvaluator:
    """
    RAGAS evaluator wrapper.

    Supports:
    - OpenAI LLM backend
    - configurable metrics
    - dataframe evaluation
    - side-by-side system comparison
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:

        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS not installed."
            )

        # ====================================================
        # API KEY
        # ====================================================

        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "Missing OpenAI API key."
            )

        # ====================================================
        # LLM
        # ====================================================

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            api_key=self.api_key,
        )

        # ====================================================
        # EMBEDDINGS
        # ====================================================

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=self.api_key,
        )

        # ====================================================
        # AVAILABLE METRICS
        # ====================================================

        self.available_metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "context_relevancy": context_relevancy,
            "answer_correctness": answer_correctness,
            "answer_similarity": answer_similarity,
        }

        # ====================================================
        # SELECTED METRICS
        # ====================================================

        if metrics is None:

            metrics = [
                "faithfulness",
                "answer_relevancy",
            ]

        invalid = [
            m for m in metrics
            if m not in self.available_metrics
        ]

        if invalid:
            raise ValueError(
                f"Invalid metrics: {invalid}"
            )

        self.metric_names = metrics

        self.metrics = [
            self.available_metrics[m]
            for m in metrics
        ]

    # ========================================================
    # DATASET PREP
    # ========================================================

    def prepare_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: Optional[List[List[str]]] = None,
        ground_truths: Optional[List[str]] = None,
    ) -> Dataset:

        data: Dict[str, List] = {
            "question": questions,
            "answer": answers,
            "contexts": (
                contexts
                if contexts is not None
                else [[] for _ in questions]
            ),
        }

        if ground_truths is not None:
            data["ground_truth"] = ground_truths

        return Dataset.from_dict(data)

    # ========================================================
    # MAIN EVALUATION
    # ========================================================

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: Optional[List[List[str]]] = None,
        ground_truths: Optional[List[str]] = None,
        show_progress: bool = True,
    ):

        dataset = self.prepare_dataset(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
        )

        if show_progress:

            logger.info(
                "Running RAGAS on %d samples",
                len(questions)
            )

        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )

        return result

    # ========================================================
    # DATAFRAME EVALUATION
    # ========================================================

    def evaluate_from_dataframe(
        self,
        df: pd.DataFrame,
        question_col: str = "question",
        answer_col: str = "predicted_answer",
        contexts_col: Optional[str] = "contexts",
        ground_truth_col: Optional[str] = "gold_answer",
        show_progress: bool = True,
    ):

        questions = (
            df[question_col]
            .fillna("")
            .astype(str)
            .tolist()
        )

        answers = (
            df[answer_col]
            .fillna("")
            .astype(str)
            .tolist()
        )

        # ====================================================
        # CONTEXT PARSING
        # ====================================================

        contexts = None

        if (
            contexts_col
            and contexts_col in df.columns
        ):

            contexts = []

            for ctx in df[contexts_col]:

                if isinstance(ctx, str):

                    try:
                        ctx = ast.literal_eval(ctx)

                    except Exception:
                        ctx = [ctx]

                elif not isinstance(ctx, list):

                    ctx = [str(ctx)]

                contexts.append(ctx)

        # ====================================================
        # GROUND TRUTH
        # ====================================================

        ground_truths = None

        if (
            ground_truth_col
            and ground_truth_col in df.columns
        ):

            ground_truths = (
                df[ground_truth_col]
                .fillna("")
                .astype(str)
                .tolist()
            )

        return self.evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
            show_progress=show_progress,
        )

    # ========================================================
    # SYSTEM COMPARISON
    # ========================================================

    def compare_systems(
        self,
        system1_df: pd.DataFrame,
        system2_df: pd.DataFrame,
        system1_name: str = "System 1",
        system2_name: str = "System 2",
        **kwargs,
    ) -> pd.DataFrame:

        logger.info(
            "Evaluating %s",
            system1_name
        )

        results1 = self.evaluate_from_dataframe(
            system1_df,
            **kwargs,
        )

        logger.info(
            "Evaluating %s",
            system2_name
        )

        results2 = self.evaluate_from_dataframe(
            system2_df,
            **kwargs,
        )

        comparison = []

        for metric in results1.keys():

            if metric not in results2:
                continue

            v1 = results1[metric]
            v2 = results2[metric]

            delta = v2 - v1

            delta_pct = (
                (delta / v1) * 100
                if v1 != 0
                else 0
            )

            comparison.append({
                "Metric": metric,
                system1_name: round(v1, 4),
                system2_name: round(v2, 4),
                "Δ": round(delta, 4),
                "Δ%": round(delta_pct, 2),
            })

        return pd.DataFrame(comparison)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def evaluate_rag_pipeline(
    predictions_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    question_col: str = "question",
    answer_col: str = "predicted_answer",
    contexts_col: str = "contexts",
    ground_truth_col: Optional[str] = "gold_answer",
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
):

    evaluator = RAGASEvaluator(
        metrics=metrics,
        llm_model=llm_model,
        embedding_model=embedding_model,
        api_key=api_key,
    )

    return evaluator.evaluate_from_dataframe(
        predictions_df,
        question_col=question_col,
        answer_col=answer_col,
        contexts_col=contexts_col,
        ground_truth_col=ground_truth_col,
    )


def compare_rag_systems(
    system1_df: pd.DataFrame,
    system2_df: pd.DataFrame,
    system1_name: str = "Normal RAG",
    system2_name: str = "Filtered RAG",
    metrics: Optional[List[str]] = None,
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    **kwargs,
):

    evaluator = RAGASEvaluator(
        metrics=metrics,
        llm_model=llm_model,
        embedding_model=embedding_model,
        api_key=api_key,
    )

    return evaluator.compare_systems(
        system1_df=system1_df,
        system2_df=system2_df,
        system1_name=system1_name,
        system2_name=system2_name,
        **kwargs,
    )