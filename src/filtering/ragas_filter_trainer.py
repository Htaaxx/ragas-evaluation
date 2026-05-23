"""
ragas_filter_trainer.py

Train sklearn classifiers from RAGAS feature files.

Expected feature data:
- label column
- RAGAS numeric feature columns, e.g. faithfulness, answer_relevancy, context_relevancy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .helper import (
    DEFAULT_RAGAS_FEATURE_COLS,
    _ensure_path,
    _normalize_col_aliases,
    _safe_json_dump,
    get_default_models,
)


@dataclass
class TrainResult:
    results_df: pd.DataFrame
    best_model_name: str
    best_model: BaseEstimator
    feature_cols: List[str]
    test_predictions: pd.DataFrame


class RagasFilterTrainer:
    """Train sklearn classifiers from RAGAS feature DataFrame/file."""

    def __init__(
        self,
        feature_data: Optional[Union[str, Path, pd.DataFrame]] = None,
        output_dir: Union[str, Path] = "./models/ragas_filter",
        label_col: str = "label",
        id_col: str = "id",
        feature_cols: Optional[Sequence[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        sort_by: str = "f1",
        models: Optional[Dict[str, BaseEstimator]] = None,
    ) -> None:
        self.feature_data = feature_data
        self.output_dir = _ensure_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_col = label_col
        self.id_col = id_col
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.test_size = test_size
        self.random_state = random_state
        self.sort_by = sort_by
        self.models = models or get_default_models(random_state=random_state)

        self.df: Optional[pd.DataFrame] = None
        self.results_df: Optional[pd.DataFrame] = None
        self.trained_models: Dict[str, BaseEstimator] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[BaseEstimator] = None

    def load_features(self, feature_data: Optional[Union[str, Path, pd.DataFrame]] = None) -> pd.DataFrame:
        data = feature_data if feature_data is not None else self.feature_data
        if data is None:
            raise ValueError("feature_data is required.")
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            path = _ensure_path(data)
            if not path.exists():
                raise FileNotFoundError(f"Feature file not found: {path}")
            suffix = path.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(path)
            elif suffix == ".jsonl":
                df = pd.read_json(path, lines=True)
            elif suffix == ".json":
                df = pd.read_json(path)
            elif suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported feature file type: {suffix}")
        df = _normalize_col_aliases(df)
        if self.label_col not in df.columns:
            raise ValueError(f"Missing label column: {self.label_col}")
        self.df = df
        return df

    def infer_feature_cols(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        df = self.df if df is None else df
        if df is None:
            raise ValueError("No feature DataFrame loaded.")
        if self.feature_cols is not None:
            missing = [c for c in self.feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing requested feature columns: {missing}")
            return list(self.feature_cols)

        cols = [c for c in DEFAULT_RAGAS_FEATURE_COLS if c in df.columns]
        if cols:
            return cols

        exclude = {self.label_col, self.id_col, "filter_label", "filter_confidence"}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in numeric_cols if c not in exclude]
        if not cols:
            raise ValueError("Could not infer any numeric feature columns.")
        return cols

    def split_data(self, df: pd.DataFrame, feature_cols: Sequence[str]):
        X = df[list(feature_cols)]
        y = df[self.label_col].astype(int)
        stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

    def evaluate_classifier(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[Dict[str, float], np.ndarray, Optional[np.ndarray]]:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, probabilities)
            except Exception:
                metrics["roc_auc"] = np.nan
        else:
            metrics["roc_auc"] = np.nan
        return metrics, predictions, probabilities

    def train_all(self, df: Optional[pd.DataFrame] = None) -> TrainResult:
        if df is None:
            df = self.load_features()
        feature_cols = self.infer_feature_cols(df)
        X_train, X_test, y_train, y_test = self.split_data(df, feature_cols)

        results = []
        test_pred_rows = []
        self.trained_models = {}

        for name, model in self.models.items():
            fitted_model = clone(model)
            metrics, preds, probs = self.evaluate_classifier(
                fitted_model, X_train, X_test, y_train, y_test
            )
            results.append({"model": name, **metrics})
            self.trained_models[name] = fitted_model

            pred_df = pd.DataFrame({"model": name, "y_true": y_test.values, "y_pred": preds})
            if probs is not None:
                pred_df["y_prob"] = probs
            if self.id_col in df.columns:
                pred_df[self.id_col] = df.loc[y_test.index, self.id_col].values
            test_pred_rows.append(pred_df)

        results_df = pd.DataFrame(results).sort_values(
            [self.sort_by, "accuracy"], ascending=False
        ).reset_index(drop=True)
        self.results_df = results_df
        self.best_model_name = str(results_df.iloc[0]["model"])
        self.best_model = self.trained_models[self.best_model_name]

        test_predictions = pd.concat(test_pred_rows, ignore_index=True)
        return TrainResult(
            results_df=results_df,
            best_model_name=self.best_model_name,
            best_model=self.best_model,
            feature_cols=list(feature_cols),
            test_predictions=test_predictions,
        )

    def save_model(
        self,
        model: Optional[BaseEstimator] = None,
        model_name: Optional[str] = None,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> Path:
        model = model or self.best_model
        model_name = model_name or self.best_model_name
        if model is None or model_name is None:
            raise ValueError("No trained model found. Run train_all() first.")
        feature_cols = list(feature_cols or self.infer_feature_cols(self.df))
        bundle = {
            "model": model,
            "model_name": model_name,
            "feature_cols": feature_cols,
            "label_col": self.label_col,
            "id_col": self.id_col,
        }
        save_path = self.output_dir / f"{model_name}.joblib"
        joblib.dump(bundle, save_path)
        return save_path

    def save_reports(self, train_result: TrainResult) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}
        results_path = self.output_dir / "training_results.csv"
        train_result.results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
        paths["training_results"] = results_path

        preds_path = self.output_dir / "test_predictions.csv"
        train_result.test_predictions.to_csv(preds_path, index=False, encoding="utf-8-sig")
        paths["test_predictions"] = preds_path

        meta_path = self.output_dir / "training_metadata.json"
        _safe_json_dump({
            "best_model_name": train_result.best_model_name,
            "feature_cols": train_result.feature_cols,
            "sort_by": self.sort_by,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "models": list(self.models.keys()),
        }, meta_path)
        paths["training_metadata"] = meta_path
        return paths

    def run(self) -> Dict[str, Any]:
        df = self.load_features()
        train_result = self.train_all(df)
        model_path = self.save_model(
            model=train_result.best_model,
            model_name=train_result.best_model_name,
            feature_cols=train_result.feature_cols,
        )
        report_paths = self.save_reports(train_result)
        return {
            "results_df": train_result.results_df,
            "best_model_name": train_result.best_model_name,
            "best_model": train_result.best_model,
            "feature_cols": train_result.feature_cols,
            "model_path": model_path,
            "report_paths": report_paths,
            "test_predictions": train_result.test_predictions,
        }

def train_ragas_filter(
    feature_path: Union[str, Path],
    output_dir: Union[str, Path] = "./models/ragas_filter",
    label_col: str = "label",
    sort_by: str = "f1",
) -> Dict[str, Any]:
    trainer = RagasFilterTrainer(
        feature_data=feature_path,
        output_dir=output_dir,
        label_col=label_col,
        sort_by=sort_by,
    )
    return trainer.run()
