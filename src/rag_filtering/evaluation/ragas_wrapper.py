"""
ragas_wrapper.py

RAGAS wrapper used for feature extraction by the RAGAS faithfulness filter.

Supports:
- an OpenAI-compatible LLM + embedding backend (configurable)
- per-cell retry of failed RAGAS metrics
- resumable checkpoint evaluation for large datasets

This is intentionally separate from ``ragas_evaluator.RAGASEvaluator`` (which is
a lighter black-box answer evaluator). Feature extraction consumes the ``RAGAS``
class here because it exposes ``evaluate`` / ``evaluate_checkpoint`` returning a
``.to_pandas()``-able result.

Example
-------
    evaluator = RAGAS(
        metrics=["faithfulness", "answer_relevancy"],
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    result = evaluator.evaluate(questions, answers, contexts)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        answer_similarity,
        context_precision,
        context_recall,
        faithfulness,
    )

    RAGAS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional heavy deps
    RAGAS_AVAILABLE = False
    logger.warning(
        "RAGAS dependencies missing. Install with: "
        "pip install ragas langchain-openai datasets"
    )


def _default_available_metrics() -> Dict[str, object]:
    """Build the metric registry, skipping metrics removed in newer RAGAS."""
    if not RAGAS_AVAILABLE:
        return {}
    metrics = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
        "answer_similarity": answer_similarity,
    }
    try:
        from ragas.metrics import context_relevancy  # deprecated in ragas>=0.4

        metrics["context_relevancy"] = context_relevancy
    except ImportError:
        logger.info(
            "context_relevancy unavailable in this RAGAS version; skipping."
        )
    return metrics


class SelfEvaluationResult:
    """Lightweight result wrapper exposing ``to_pandas`` + a metric summary."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.summary = {
            col: df[col].mean()
            for col in df.columns
            if col
            not in ["sample_idx", "question", "answer", "contexts", "ground_truth"]
        }

    def to_pandas(self) -> pd.DataFrame:
        return self.df

    def __getitem__(self, item):
        return self.summary[item]

    def keys(self):
        return self.summary.keys()

    def __repr__(self) -> str:
        return str(self.summary)


class RAGAS:
    """RAGAS wrapper with an OpenAI backend, retry, and checkpointing."""

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
                "RAGAS not installed. "
                "pip install ragas langchain-openai datasets"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key (set OPENAI_API_KEY).")

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            api_key=self.api_key,
        )
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=self.api_key,
        )

        self.available_metrics = _default_available_metrics()

        if metrics is None:
            metrics = ["faithfulness", "answer_relevancy"]
        invalid = [m for m in metrics if m not in self.available_metrics]
        if invalid:
            raise ValueError(
                f"Invalid or unavailable metrics: {invalid}. "
                f"Available: {sorted(self.available_metrics)}"
            )

        self.metric_names = metrics
        self.metrics = [self.available_metrics[m] for m in metrics]

    def _is_failed_value(self, value) -> bool:
        if value is None:
            return True
        try:
            if pd.isna(value):
                return True
        except Exception:
            pass
        if isinstance(value, str):
            return value.strip().lower() in {"none", "nan", "null", ""}
        return False

    def _find_failed_cells(self, result_df: pd.DataFrame):
        failed = []
        metric_names = [getattr(m, "name", None) for m in self.metrics]
        metric_names = [n for n in metric_names if n in result_df.columns]
        for row_idx, row in result_df.iterrows():
            for metric_name in metric_names:
                if self._is_failed_value(row[metric_name]):
                    failed.append((row_idx, metric_name))
        return failed

    def prepare_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: Optional[List[List[str]]] = None,
        ground_truths: Optional[List[str]] = None,
    ) -> "Dataset":
        data: Dict[str, List] = {
            "question": questions,
            "answer": answers,
            "contexts": contexts if contexts is not None else [[] for _ in questions],
        }
        if ground_truths is not None:
            data["ground_truth"] = ground_truths
        return Dataset.from_dict(data)

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: Optional[List[List[str]]] = None,
        ground_truths: Optional[List[str]] = None,
        show_progress: bool = True,
        max_retries: int = 3,
    ):
        dataset = self.prepare_dataset(questions, answers, contexts, ground_truths)
        if show_progress:
            logger.info("Running RAGAS on %d samples", len(questions))

        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )
        result_df = result.to_pandas()
        failed_cells = self._find_failed_cells(result_df)
        if not failed_cells:
            return result

        if show_progress:
            logger.warning(
                "Found %d failed RAGAS cells. Retrying ...", len(failed_cells)
            )

        for row_idx, metric_name in failed_cells:
            for attempt in range(1, max_retries + 1):
                metric = next(
                    m for m in self.metrics if getattr(m, "name", None) == metric_name
                )
                single_dataset = self.prepare_dataset(
                    questions=[questions[row_idx]],
                    answers=[answers[row_idx]],
                    contexts=[contexts[row_idx]] if contexts is not None else None,
                    ground_truths=(
                        [ground_truths[row_idx]] if ground_truths is not None else None
                    ),
                )
                retry_result = evaluate(
                    dataset=single_dataset,
                    metrics=[metric],
                    llm=self.llm,
                    embeddings=self.embeddings,
                )
                new_value = retry_result.to_pandas().loc[0, metric_name]
                if not self._is_failed_value(new_value):
                    result_df.loc[row_idx, metric_name] = new_value
                    if show_progress:
                        logger.info(
                            "Repaired RAGAS cell row=%s metric=%s attempt=%s",
                            row_idx,
                            metric_name,
                            attempt,
                        )
                    break
        return SelfEvaluationResult(result_df)

    def evaluate_checkpoint(
        self,
        questions: List[str],
        answers: List[str],
        contexts: Optional[List[List[str]]] = None,
        ground_truths: Optional[List[str]] = None,
        batch_size: int = 10,
        save_path: str = "ragas_checkpoint.csv",
        show_progress: bool = True,
    ) -> SelfEvaluationResult:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        all_results: List[dict] = []
        start_idx = 0
        if save_path.exists():
            checkpoint_df = pd.read_csv(save_path)
            all_results = checkpoint_df.to_dict("records")
            start_idx = len(checkpoint_df)
            logger.info("Resuming RAGAS checkpoint from sample %d", start_idx)

        n = len(questions)
        if contexts is None:
            contexts = [[] for _ in questions]

        for start in range(start_idx, n, batch_size):
            end = min(start + batch_size, n)
            batch_ground_truths = (
                ground_truths[start:end] if ground_truths is not None else None
            )
            result = self.evaluate(
                questions=questions[start:end],
                answers=answers[start:end],
                contexts=contexts[start:end],
                ground_truths=batch_ground_truths,
                show_progress=show_progress,
            )
            result_df = result.to_pandas()
            result_df.insert(0, "sample_idx", range(start, end))
            all_results.extend(result_df.to_dict("records"))
            pd.DataFrame(all_results).to_csv(save_path, index=False)
            logger.info("Saved RAGAS checkpoint: %d/%d", end, n)

        return SelfEvaluationResult(pd.DataFrame(all_results))
