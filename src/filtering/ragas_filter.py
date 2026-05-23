"""
ragas_filter.py

RagasFilter: Raw data + feature/model -> filter predictions + evaluation

------------------------
Required:
    - id
    - question
    - answer
    - context

Optional:
    - label
    - gold_ans / gold_answer / reference / reference_answer

Label convention:
    - 1 = accepted / faithful / grounded
    - 0 = rejected / hallucinated / unsupported
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
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

from .helper import (
    DEFAULT_RAGAS_FEATURE_COLS,
    GOLD_ANSWER_CANDIDATES,
    _ensure_path,
    _normalize_col_aliases,
    _get_gold_col,
    _safe_json_dump,
    parse_context,
)

from .ragas_feature_extractor import RagasFeatureExtractor



class RagasFilter:
    """
    Predict + evaluate using trained RAGAS filter model.

    Input can still be raw data so evaluation can use label/gold_ans/context.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        feature_extractor: Optional[RagasFeatureExtractor] = None,
        output_dir: Union[str, Path] = "./results/inference",
        id_col: str = "id",
        label_col: str = "label",
        answer_col: str = "answer",
        context_col: str = "context",
        gold_col: Optional[str] = None,
        threshold: float = 0.5,
    ) -> None:
        self.model_path = _ensure_path(model_path)
        self.feature_extractor = feature_extractor
        self.output_dir = _ensure_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.id_col = id_col
        self.label_col = label_col
        self.answer_col = answer_col
        self.context_col = context_col
        self.gold_col = gold_col
        self.threshold = threshold

        self.model: Optional[BaseEstimator] = None
        self.model_name: Optional[str] = None
        self.feature_cols: Optional[List[str]] = None
        self.feature_df: Optional[pd.DataFrame] = None
        self.output_df: Optional[pd.DataFrame] = None

    def load_model(self) -> BaseEstimator:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        obj = joblib.load(self.model_path)
        if isinstance(obj, dict) and "model" in obj:
            self.model = obj["model"]
            self.model_name = obj.get("model_name")
            self.feature_cols = obj.get("feature_cols")
        else:
            self.model = obj
            self.model_name = self.model_path.stem
            self.feature_cols = None
        return self.model

    def prepare_features(
        self,
        data: Optional[Union[str, Path, pd.DataFrame]] = None,
        feature_df: Optional[pd.DataFrame] = None,
        feature_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        batch_size: int = 50,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        if feature_df is not None:
            df = _normalize_col_aliases(feature_df.copy())
        elif feature_path is not None and _ensure_path(feature_path).exists():
            df = _normalize_col_aliases(pd.read_csv(feature_path))
        else:
            if data is None:
                raise ValueError("Either feature_df, feature_path, or data is required.")
            if self.feature_extractor is None:
                raise ValueError("feature_extractor is required when feature_df/feature_path is not provided.")
            df = self.feature_extractor.transform(
                data=data,
                feature_path=feature_path,
                checkpoint_path=checkpoint_path,
                batch_size=batch_size,
                show_progress=show_progress,
            )
            df = _normalize_col_aliases(df)
        self.feature_df = df
        return df

    def predict(
        self,
        data: Optional[Union[str, Path, pd.DataFrame]] = None,
        feature_df: Optional[pd.DataFrame] = None,
        feature_path: Optional[Union[str, Path]] = None,
        filter_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        batch_size: int = 50,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        if self.model is None:
            self.load_model()
        assert self.model is not None

        df = self.prepare_features(
            data=data,
            feature_df=feature_df,
            feature_path=feature_path,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        if self.feature_cols is None:
            self.feature_cols = [c for c in DEFAULT_RAGAS_FEATURE_COLS if c in df.columns]
            if not self.feature_cols:
                exclude = {self.label_col, self.id_col, "filter_label", "filter_confidence"}
                self.feature_cols = [
                    c for c in df.select_dtypes(include=[np.number]).columns.tolist()
                    if c not in exclude
                ]
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns for prediction: {missing}")

        X = df[self.feature_cols]
        if hasattr(self.model, "predict_proba"):
            try:
                prob = self.model.predict_proba(X)[:, 1]
                pred = (prob >= self.threshold).astype(int)
            except Exception:
                pred = self.model.predict(X).astype(int)
                prob = pred.astype(float)
        else:
            pred = self.model.predict(X).astype(int)
            prob = pred.astype(float)

        pred_df = pd.DataFrame({
            self.id_col: df[self.id_col].values if self.id_col in df.columns else np.arange(len(df)),
            "filter_label": pred.astype(int),
            "filter_confidence": prob.astype(float),
        })
        merged = pd.concat(
            [df.reset_index(drop=True), pred_df.drop(columns=[self.id_col], errors="ignore").reset_index(drop=True)],
            axis=1,
        )
        self.output_df = merged

        if filter_path is not None:
            filter_path = _ensure_path(filter_path)
            filter_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(filter_path, index=False, encoding="utf-8-sig")
        return merged

    def evaluate_classification(
        self,
        df: Optional[pd.DataFrame] = None,
        label_col: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        df = df if df is not None else self.output_df
        if df is None:
            raise ValueError("No prediction output found. Run predict() first.")
        label_col = label_col or self.label_col
        if label_col not in df.columns:
            raise ValueError(f"Cannot evaluate classification: missing label column `{label_col}`.")

        y_true = df[label_col].astype(int)
        y_pred = df["filter_label"].astype(int)
        metrics: Dict[str, Any] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
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
        df: Optional[pd.DataFrame] = None,
        evaluator: Optional[Any] = None,
        gold_col: Optional[str] = None,
        answer_col: Optional[str] = None,
        context_col: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        df = df if df is not None else self.output_df
        if df is None:
            raise ValueError("No prediction output found. Run predict() first.")

        answer_col = answer_col or self.answer_col
        context_col = context_col or self.context_col
        gold_col = gold_col or self.gold_col or _get_gold_col(df)
        if gold_col is None or gold_col not in df.columns:
            raise ValueError(f"Cannot evaluate answer quality: missing gold answer column. Tried {GOLD_ANSWER_CANDIDATES}")
        if answer_col not in df.columns:
            raise ValueError(f"Missing answer column: {answer_col}")

        accepted_df = df[df["filter_label"].astype(int) == 1].copy()

        if evaluator is None:
            try:
                from src.evaluation.evaluator import TraditionalEvaluator
                evaluator = TraditionalEvaluator(metrics=metrics or [
                    "str_em", "rouge", "mauve", "citation_precision", "citation_recall"
                ])
            except Exception as exc:
                raise ImportError("Could not import TraditionalEvaluator. Pass evaluator manually.") from exc

        if hasattr(evaluator, "compare_filtered_quality"):
            comparison = evaluator.compare_filtered_quality(
                df=df,
                accepted_df=accepted_df,
                prediction_col=answer_col,
                reference_col=gold_col,
                context_col=context_col,
            )
        else:
            full_scores = _evaluate_with_traditional_evaluator(evaluator, df, answer_col, gold_col, context_col)
            accepted_scores = _evaluate_with_traditional_evaluator(evaluator, accepted_df, answer_col, gold_col, context_col)
            rows = []
            for metric in sorted(set(full_scores) | set(accepted_scores)):
                before = full_scores.get(metric, np.nan)
                after = accepted_scores.get(metric, np.nan)
                rows.append({
                    "metric": metric,
                    "unfiltered": before,
                    "accepted": after,
                    "delta": after - before if pd.notna(before) and pd.notna(after) else np.nan,
                })
            comparison = pd.DataFrame(rows)

        coverage = pd.DataFrame([
            {"metric": "num_samples", "unfiltered": len(df), "accepted": len(accepted_df), "delta": len(accepted_df) - len(df)},
            {"metric": "acceptance_rate", "unfiltered": 1.0, "accepted": len(accepted_df) / max(len(df), 1), "delta": len(accepted_df) / max(len(df), 1) - 1.0},
        ])
        comparison = pd.concat([coverage, comparison], ignore_index=True)

        if save_path is not None:
            save_path = _ensure_path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            comparison.to_csv(save_path, index=False, encoding="utf-8-sig")
        return comparison

    def evaluate(
        self,
        df: Optional[pd.DataFrame] = None,
        mode: str = "both",
        evaluator: Optional[Any] = None,
        data_name: str = "data",    
    ) -> Dict[str, Any]:
        df = df if df is not None else self.output_df
        if df is None:
            raise ValueError("No prediction output found. Run predict() first.")
        result: Dict[str, Any] = {}

        if mode in ["classification", "both"]:
            if self.label_col in df.columns:
                result["classification"] = self.evaluate_classification(
                    df=df,
                    save_path=self.output_dir / f"{data_name}_classification.json",
                )
            else:
                logger.warning("Skipping classification evaluation: no label column found.")

        if mode in ["quality", "both"]:
            if _get_gold_col(df) is not None or self.gold_col is not None:
                result["quality"] = self.evaluate_answer_quality(
                    df=df,
                    evaluator=evaluator,
                    save_path=self.output_dir / f"{data_name}_quality.csv",
                )
            else:
                logger.warning("Skipping answer quality evaluation: no gold answer column found.")
        return result

    def run(
        self,
        data: Optional[Union[str, Path, pd.DataFrame]] = None,
        feature_df: Optional[pd.DataFrame] = None,
        feature_path: Optional[Union[str, Path]] = None,
        filter_path: Optional[Union[str, Path]] = None,
        data_name: str = "data",
        checkpoint_path: Optional[Union[str, Path]] = None,
        evaluator: Optional[Any] = None,
        eval_mode: str = "both",
        batch_size: int = 50,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        output_df = self.predict(
            data=data,
            feature_df=feature_df,
            feature_path=feature_path,
            filter_path=filter_path,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        evaluation = self.evaluate(df=output_df, mode=eval_mode, evaluator=evaluator, data_name=data_name)
        return {"output_df": output_df, "evaluation": evaluation}


def _evaluate_with_traditional_evaluator(
    evaluator: Any,
    df: pd.DataFrame,
    prediction_col: str,
    reference_col: str,
    context_col: Optional[str] = None,
) -> Dict[str, float]:
    if len(df) == 0:
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
    except AttributeError:
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
        contexts = df[context_col].apply(parse_context).tolist() if context_col and context_col in df.columns else None
        return evaluator.evaluate(predictions=predictions, references=references, contexts=contexts)
    except TypeError:
        return evaluator.evaluate(predictions=predictions, references=references)



def run_ragas_filter(
    data_path: Union[str, Path],
    model_path: Union[str, Path],
    ragas_evaluator: Optional[Any] = None,
    feature_path: Optional[Union[str, Path]] = None,
    filter_path: Optional[Union[str, Path]] = None,
    evaluator: Optional[Any] = None,
    eval_mode: str = "both",
) -> Dict[str, Any]:
    feature_extractor = RagasFeatureExtractor(ragas_evaluator=ragas_evaluator) if ragas_evaluator is not None else None
    ragas_filter = RagasFilter(model_path=model_path, feature_extractor=feature_extractor)
    return ragas_filter.run(
        data=data_path,
        feature_path=feature_path,
        filter_path=filter_path,
        evaluator=evaluator,
        eval_mode=eval_mode,
    )
