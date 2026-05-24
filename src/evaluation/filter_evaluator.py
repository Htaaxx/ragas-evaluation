"""
filter_evaluator.py

Shared evaluator for answer filters.

Supports two evaluation modes:
1. Classification quality
   - requires label column
   - compares label vs filter_label

2. Answer quality after filtering
   - requires gold answer column
   - compares unfiltered full set vs accepted set

Expected prediction dataframe columns:
    - id
    - question
    - answer
    - context
    - filter_label
    - filter_confidence optional
    - label optional
    - gold_ans / gold_answer / reference / reference_answer optional
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from .evaluator import TraditionalEvaluator
from ..filtering.helper import GOLD_ANSWER_CANDIDATES, _ensure_path, _safe_json_dump, _get_gold_col

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class FilterEvaluator:
    """
    Shared evaluation class for all filter modules.

    Label convention:
        1 = accepted / faithful / grounded
        0 = rejected / hallucinated / unsupported
    """

    def __init__(
        self,
        label_col: str = "label",
        answer_col: str = "answer",
        context_col: str = "context",
        gold_col: Optional[str] = None,
        output_dir: Union[str, Path] = "./results/filter_eval",
    ) -> None:
        self.label_col = label_col
        self.answer_col = answer_col
        self.context_col = context_col
        self.gold_col = gold_col
        self.output_dir = _ensure_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_classification(
        self,
        df: pd.DataFrame,
        label_col: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        label_col = label_col or self.label_col

        if label_col not in df.columns:
            raise ValueError(f"Cannot evaluate classification: missing label column `{label_col}`.")

        if "filter_label" not in df.columns:
            raise ValueError("Cannot evaluate classification: missing `filter_label` column.")

        y_true = df[label_col].astype(int)
        y_pred = df["filter_label"].astype(int)

        metrics: Dict[str, Any] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
        }

        if "filter_confidence" in df.columns and y_true.nunique() == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, df["filter_confidence"])
            except Exception:
                metrics["roc_auc"] = np.nan

        if save_path is not None:
            _safe_json_dump(metrics, save_path)

        return metrics

    def evaluate_answer_quality(
        self,
        df: pd.DataFrame,
        evaluator: Optional[TraditionalEvaluator] = None,
        gold_col: Optional[str] = None,
        answer_col: Optional[str] = None,
        context_col: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Compare answer quality of:
            - full unfiltered dataset
            - accepted subset where filter_label == 1
        """
        if "filter_label" not in df.columns:
            raise ValueError("Cannot evaluate answer quality: missing `filter_label` column.")

        answer_col = answer_col or self.answer_col
        context_col = context_col or self.context_col
        gold_col = gold_col or self.gold_col or _get_gold_col(df)

        if answer_col not in df.columns:
            raise ValueError(f"Missing answer column `{answer_col}`.")

        if gold_col is None or gold_col not in df.columns:
            raise ValueError(
                "Cannot evaluate answer quality: missing gold answer column. "
                f"Tried: {GOLD_ANSWER_CANDIDATES}"
            )

        accepted_df = df[df["filter_label"].astype(int) == 1].copy()

        if evaluator is None:
            try:
                from src.evaluation.evaluator import TraditionalEvaluator

                evaluator = TraditionalEvaluator(
                    metrics=[
                        "str_em",
                        "rouge",
                        "mauve",
                        "citation_precision",
                        "citation_recall",
                    ]
                )
            except Exception as exc:
                raise ImportError(
                    "Could not import TraditionalEvaluator. "
                    "Pass evaluator=TraditionalEvaluator(...) manually."
                ) from exc

        if hasattr(evaluator, "compare_filtered_quality"):
            comparison_df = evaluator.compare_filtered_quality(
                df=df,
                accepted_df=accepted_df,
                prediction_col=answer_col,
                reference_col=gold_col,
                context_col=context_col,
            )
        else:
            full_scores = self._evaluate_with_traditional_evaluator(
                evaluator=evaluator,
                df=df,
                prediction_col=answer_col,
                reference_col=gold_col,
                context_col=context_col,
            )
            accepted_scores = self._evaluate_with_traditional_evaluator(
                evaluator=evaluator,
                df=accepted_df,
                prediction_col=answer_col,
                reference_col=gold_col,
                context_col=context_col,
            )

            rows = []
            for metric in sorted(set(full_scores) | set(accepted_scores)):
                before = full_scores.get(metric, np.nan)
                after = accepted_scores.get(metric, np.nan)
                rows.append(
                    {
                        "metric": metric,
                        "unfiltered": before,
                        "accepted": after,
                        "delta": (
                            after - before
                            if pd.notna(before) and pd.notna(after)
                            else np.nan
                        ),
                    }
                )

            comparison_df = pd.DataFrame(rows)

        coverage_df = pd.DataFrame(
            [
                {
                    "metric": "num_samples",
                    "unfiltered": len(df),
                    "accepted": len(accepted_df),
                    "delta": len(accepted_df) - len(df),
                },
                {
                    "metric": "acceptance_rate",
                    "unfiltered": 1.0,
                    "accepted": len(accepted_df) / max(len(df), 1),
                    "delta": len(accepted_df) / max(len(df), 1) - 1.0,
                },
            ]
        )

        comparison_df = pd.concat([coverage_df, comparison_df], ignore_index=True)

        if save_path is not None:
            save_path = _ensure_path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(save_path, index=False, encoding="utf-8-sig")

        return comparison_df

    def evaluate(
        self,
        df: pd.DataFrame,
        mode: str = "both",
        evaluator: Optional[TraditionalEvaluator] = None,
        output_prefix: str = "filter",
    ) -> Dict[str, Any]:
        """
        mode:
            - classification
            - quality
            - both
        """
        results: Dict[str, Any] = {}

        if mode in ["classification", "both"]:
            if self.label_col in df.columns:
                results["classification"] = self.evaluate_classification(
                    df=df,
                    save_path=self.output_dir / f"{output_prefix}_classification.json",
                )
            else:
                logger.warning("Skip classification eval: no `%s` column.", self.label_col)

        if mode in ["quality", "both"]:
            gold_col = self.gold_col or _get_gold_col(df)
            if gold_col is not None:
                results["quality"] = self.evaluate_answer_quality(
                    df=df,
                    evaluator=evaluator,
                    gold_col=gold_col,
                    save_path=self.output_dir / f"{output_prefix}_quality.csv",
                )
            else:
                logger.warning("Skip answer quality eval: no gold answer column.")

        return results

    @staticmethod
    def _evaluate_with_traditional_evaluator(
        evaluator: Optional[TraditionalEvaluator],
        df: pd.DataFrame,
        prediction_col: str,
        reference_col: str,
        context_col: Optional[str] = None,
    ) -> Dict[str, float]:
        if len(df) == 0:
            return {}

        if evaluator is None:
            return {}

        try:
            return evaluator.evaluate_from_dataframe(
                df=df,
                prediction_col=prediction_col,
                reference_col=reference_col,
                context_col=context_col,
            )
        except TypeError:
            pass

        try:
            return evaluator.evaluate_from_dataframe(
                df=df,
                prediction_col=prediction_col,
                reference_col=reference_col,
            )
        except AttributeError:
            pass

        predictions = df[prediction_col].fillna("").astype(str).tolist()
        references = df[reference_col].fillna("").astype(str).tolist()

        try:
            contexts = df[context_col].tolist() if context_col and context_col in df.columns else None
            return evaluator.evaluate(
                predictions=predictions,
                references=references,
                contexts=contexts,
            )
        except TypeError:
            return evaluator.evaluate(
                predictions=predictions,
                references=references,
            )


def plot_evaluation_results(evaluation, data_name):
    if "classification" in evaluation:
        
        metrics = evaluation["classification"]

        print(f"\nClassification Metrics for {data_name}:")
        for key, value in metrics.items():
            if key in {"accuracy", "precision", "recall", "f1", "roc_auc"}:
                print(f"  {key}: {value}")


        cm = np.array(metrics["confusion_matrix"])

        plt.figure(figsize=(4,3))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Rejected", "Accepted"],
            yticklabels=["Rejected", "Accepted"],
        )

        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix")

        plt.show()

        report_df = pd.DataFrame(metrics["classification_report"]).transpose()
        display(report_df.round(4))

    if "quality" in evaluation:
        quality_df = evaluation["quality"]
        print(f"\nAnswer Quality Scores for {data_name}:")
        display(quality_df.describe().T)