"""
ragas_filter.py

RAGAS-feature-based inference module for contextual faithfulness filtering.

Task:
    Given precomputed RAGAS features for question, context, answer:
    predict whether the answer is grounded in / supported by the context.

Output:
    - filter_label: 1 accepted, 0 rejected
    - filter_confidence: model confidence/probability if available

Default behavior:
    - Load trained model from joblib
    - Select saved feature columns
    - Predict accepted/rejected labels
    - Evaluate with shared FilterEvaluator

Expected raw data schema:
    Required:
        - id
        - question
        - answer
        - context

    Optional:
        - label
        - gold_ans / gold_answer / reference / reference_answer

Expected feature schema:
    Required:
        - id
        - RAGAS feature columns, e.g.
            - faithfulness
            - answer_relevancy
            - context_precision
            - context_recall

    Optional:
        - label
        - raw metadata columns

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

logger = logging.getLogger(__name__)

from .helper import (
    DEFAULT_RAGAS_FEATURE_COLS,
    _ensure_path,
    _normalize_col_aliases,
    parse_context,
)

from .ragas_feature_extractor import RagasFeatureExtractor
from ..evaluation.filter_evaluator import FilterEvaluator


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

    def evaluate(
        self,
        df: Optional[pd.DataFrame] = None,
        mode: str = "both",
        evaluator: Optional[Any] = None,
        output_prefix: str = "llm_judge",
    ) -> Dict[str, Any]:
        df = df if df is not None else self.output_df

        if df is None:
            raise ValueError("No prediction output found. Run predict() first.")

        filter_evaluator = FilterEvaluator(
            label_col=self.label_col,
            answer_col=self.answer_col,
            context_col=self.context_col,
            output_dir=self.output_dir,
        )

        return filter_evaluator.evaluate(
            df=df,
            mode=mode,
            evaluator=evaluator,
            output_prefix=output_prefix,
        )

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
        eval_result = self.evaluate(
            df=output_df,
            mode=eval_mode,
            evaluator=evaluator,
        )

        return {
            "output_df": output_df,
            "evaluation": eval_result,
        }


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
