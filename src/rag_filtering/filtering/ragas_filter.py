"""
ragas_filter.py

RAGAS-feature-based inference module for contextual faithfulness filtering.

Task
----
Given precomputed (or freshly extracted) RAGAS features for a
``(question, context, answer)`` triple, predict whether the answer is grounded
in / supported by the retrieved context.

Output columns
--------------
- ``filter_label``: 1 = accepted, 0 = rejected
- ``filter_confidence``: model probability of the accepted class

Evaluation
----------
Uses the thesis :class:`FilterEvaluator` (FPR-focused). The final accept/reject
threshold should be selected on a validation set via
:func:`select_threshold_min_fpr` (minimize FPR subject to recall >= floor), NOT
argmax at 0.5.

Label convention
----------------
- 1 = accepted / faithful / grounded
- 0 = rejected / hallucinated / unsupported
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from .filter_evaluator import FilterEvaluator, select_threshold_min_fpr
from .helper import (
    DEFAULT_RAGAS_FEATURE_COLS,
    _ensure_path,
    _normalize_col_aliases,
)
from .ragas_feature_extractor import RagasFeatureExtractor

logger = logging.getLogger(__name__)


class RagasFilter:
    """Predict + evaluate using a trained RAGAS filter model.

    Input can be raw data (RAGAS features are computed on the fly) or an
    already-built feature frame / file.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        feature_extractor: Optional[RagasFeatureExtractor] = None,
        output_dir: Union[str, Path] = "./results/ragas_filter",
        id_col: str = "id",
        label_col: str = "label",
        answer_col: str = "answer",
        context_col: str = "context",
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
                raise ValueError(
                    "Either feature_df, feature_path, or data is required."
                )
            if self.feature_extractor is None:
                raise ValueError(
                    "feature_extractor is required when feature_df / "
                    "feature_path is not provided."
                )
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

    def _resolve_feature_cols(self, df: pd.DataFrame) -> List[str]:
        if self.feature_cols is None:
            feature_cols = [c for c in DEFAULT_RAGAS_FEATURE_COLS if c in df.columns]
            if not feature_cols:
                exclude = {
                    self.label_col,
                    self.id_col,
                    "filter_label",
                    "filter_confidence",
                }
                feature_cols = [
                    c
                    for c in df.select_dtypes(include=[np.number]).columns.tolist()
                    if c not in exclude
                ]
            self.feature_cols = feature_cols
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns for prediction: {missing}")
        return self.feature_cols

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

        if data is not None and self.feature_extractor is not None:
            raw_df = self.feature_extractor.prepare_data(data)
            raw_cols = [c for c in raw_df.columns if c not in df.columns]
            if (
                self.id_col in df.columns
                and self.id_col in raw_df.columns
                and raw_cols
            ):
                merge_cols = raw_cols + [self.id_col]
                df = df.merge(
                    raw_df[merge_cols].drop(columns=["_parsed_context"], errors="ignore"),
                    on=self.id_col,
                    how="left",
                )

        feature_cols = self._resolve_feature_cols(df)
        X = df[feature_cols]

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

        logger.info(
            "RagasFilter predicting with threshold=%.3f on %d samples",
            self.threshold,
            len(df),
        )

        df = df.reset_index(drop=True)
        df["filter_label"] = pred.astype(int)
        df["filter_confidence"] = prob.astype(float)
        self.output_df = df

        if filter_path is not None:
            filter_path = _ensure_path(filter_path)
            filter_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filter_path, index=False, encoding="utf-8-sig")
        return df

    def select_threshold(
        self,
        df: Optional[pd.DataFrame] = None,
        min_recall: float = 0.70,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Select the min-FPR threshold on labeled data and update self.threshold."""
        df = df if df is not None else self.output_df
        if df is None:
            raise ValueError("No data available for threshold selection.")
        if self.label_col not in df.columns:
            raise ValueError(
                f"Threshold selection requires a `{self.label_col}` column."
            )
        if "filter_confidence" not in df.columns:
            raise ValueError("Run predict() before selecting a threshold.")

        result = select_threshold_min_fpr(
            confidences=df["filter_confidence"].tolist(),
            labels=df[self.label_col].astype(int).tolist(),
            min_recall=min_recall,
        )
        self.threshold = float(result["threshold"])
        if save_path is not None:
            save_path = _ensure_path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            import json

            with open(save_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, default=str)
        return result

    def evaluate(
        self,
        df: Optional[pd.DataFrame] = None,
        output_prefix: str = "ragas_filter",
    ) -> Dict[str, Any]:
        """Compute FPR-focused classification metrics vs ground-truth labels."""
        df = df if df is not None else self.output_df
        if df is None:
            raise ValueError("No prediction output found. Run predict() first.")

        results: Dict[str, Any] = {}
        if self.label_col not in df.columns:
            logger.warning(
                "Skip filter evaluation: no `%s` column (inference-only run).",
                self.label_col,
            )
            return results

        evaluator = FilterEvaluator()
        predictions = (df["filter_label"].astype(int) == 1).tolist()
        labels = df[self.label_col].astype(int).tolist()

        filter_result = evaluator.evaluate(predictions, labels)
        filter_result.save(self.output_dir / f"{output_prefix}_test_results.json")

        baseline_result = evaluator.compute_no_filter_baseline(labels)
        comparison = evaluator.compare(
            {"No Filter": baseline_result, "RAGAS Filter": filter_result},
            save_path=self.output_dir / f"{output_prefix}_comparison.json",
        )

        results["filter_result"] = filter_result.to_dict()
        results["no_filter_baseline"] = baseline_result.to_dict()
        results["comparison"] = comparison
        return results

    def run(
        self,
        data: Optional[Union[str, Path, pd.DataFrame]] = None,
        feature_df: Optional[pd.DataFrame] = None,
        feature_path: Optional[Union[str, Path]] = None,
        filter_path: Optional[Union[str, Path]] = None,
        data_name: str = "ragas_filter",
        checkpoint_path: Optional[Union[str, Path]] = None,
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
        eval_result = self.evaluate(df=output_df, output_prefix=data_name)
        return {"output_df": output_df, "evaluation": eval_result}


def run_ragas_filter(
    data_path: Union[str, Path],
    model_path: Union[str, Path],
    ragas_evaluator: Optional[Any] = None,
    feature_path: Optional[Union[str, Path]] = None,
    filter_path: Optional[Union[str, Path]] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Convenience wrapper: apply a trained RAGAS filter to a data file."""
    feature_extractor = (
        RagasFeatureExtractor(ragas_evaluator=ragas_evaluator)
        if ragas_evaluator is not None
        else None
    )
    ragas_filter = RagasFilter(
        model_path=model_path,
        feature_extractor=feature_extractor,
        threshold=threshold,
    )
    return ragas_filter.run(
        data=data_path,
        feature_path=feature_path,
        filter_path=filter_path,
    )
