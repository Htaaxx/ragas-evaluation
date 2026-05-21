"""
Complete RAGAS-based filter pipeline for RAG answer quality classification.

Main Components:
- RagasFilter: Full training pipeline (RAGAS computation + model training)
- FilterEvaluator: Inference and evaluation on new data
"""

from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .ragas import RAGAS

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class RagasFilter:
    """
    Complete RAGAS-based filter pipeline.
    
    Workflow:
    1. Load labeled data
    2. Compute RAGAS black-box metrics
    3. Train multiple classifier models
    4. Select and save best model
    5. Optionally evaluate on test data
    """

    def __init__(
        self,
        output_dir: str | Path = "./results/ragas_filter",
        model_dir: str | Path = "./models/ragas_filter",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize RAGAS filter pipeline.

        Args:
            output_dir: Directory to save results and features
            model_dir: Directory to save trained models
            test_size: Train-test split ratio
            random_state: Random seed for reproducibility
            feature_df: Pre-computed feature DataFrame
            input_df: Input DataFrame for evaluation
        """
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.test_size = test_size
        self.random_state = random_state

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # State variables
        self.df = None  # Original labeled data
        self.feature_df = None  # Data with RAGAS features
        self.ragas_df = None  # RAGAS metrics only
        self.models = {}  # Trained models
        self.best_model = None
        self.best_model_name = None
        self.feature_cols = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    # ========================================================================
    # STEP 1: DATA LOADING & PREPARATION
    # ========================================================================

    def load_data(self, csv_path: str | Path) -> pd.DataFrame:
        """Load labeled data from CSV."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.df["label"] = self.df["label"].astype(int)

        # Verify required columns
        required = {"id", "question", "answer", "context", "supporting_facts", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        print(f"✓ Loaded {len(self.df)} samples from {csv_path}")
        return self.df

    @staticmethod
    def _safe_literal_eval(raw_value):
        """Safely parse Python literal strings."""
        if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
            return None
        if isinstance(raw_value, (dict, list)):
            return raw_value
        if not isinstance(raw_value, str):
            return raw_value
        try:
            return ast.literal_eval(raw_value.replace('""', '"'))
        except Exception:
            return raw_value

    def extract_contexts(self) -> pd.DataFrame:
        """Extract contexts for RAGAS evaluation."""
        def parse_supporting_facts(sf_raw, ctx_raw):
            try:
                sf = self._safe_literal_eval(sf_raw)
                ctx = self._safe_literal_eval(ctx_raw)

                if not isinstance(sf, dict) or not isinstance(ctx, dict):
                    return []

                sf_titles = set(sf.get("title", []))
                ctx_titles = ctx.get("title", [])
                ctx_sentences = ctx.get("sentences", [])

                facts = []
                for title, sents in zip(ctx_titles, ctx_sentences):
                    if title in sf_titles or str(title).startswith("QA"):
                        if sents:
                            facts.append(" ".join(sents))

                return facts
            except Exception:
                return []

        self.df["ragas_contexts"] = self.df.apply(
            lambda row: parse_supporting_facts(row["supporting_facts"], row["context"]),
            axis=1,
        )

        print(f"✓ Extracted contexts for {len(self.df)} samples")
        return self.df

    # ========================================================================
    # STEP 2: RAGAS METRICS COMPUTATION
    # ========================================================================

    def compute_ragas_features(self, ragas_evaluator: RAGAS) -> pd.DataFrame:
        """
        Compute RAGAS metrics and prepare feature table.

        Args:
            ragas_evaluator: RAGAS instance with metrics configured

        Returns:
            DataFrame with RAGAS features
        """
        print("\n[RAGAS Computation]")

        ragas_result = ragas_evaluator.evaluate_checkpoint(
            questions=self.df["question"].tolist(),
            answers=self.df["answer"].tolist(),
            contexts=self.df["ragas_contexts"].tolist(),
            batch_size=25,
            save_path=self.output_dir / "ragas_checkpoints.csv",
            show_progress=True,
        )

        self.ragas_df = ragas_result.to_pandas()

        # Keep only RAGAS metrics
        metric_cols = [
            col for col in self.ragas_df.columns
            if col in ["faithfulness", "answer_relevancy", "context_relevancy"]
        ]
        self.ragas_df = self.ragas_df[metric_cols]

        print(f"✓ Computed RAGAS metrics")
        print(f"\n{self.ragas_df.describe().T}")

        # Prepare feature table
        self.feature_df = pd.concat(
            [
                self.df[["id", "label"]].reset_index(drop=True),
                self.ragas_df.reset_index(drop=True),
            ],
            axis=1,
        )

        self.feature_df = self.feature_df.drop_duplicates(subset=["id"]).reset_index(
            drop=True
        )

        # Save
        feature_path = self.output_dir / "ragas_features.csv"
        self.feature_df.to_csv(feature_path, index=False)
        print(f"✓ Saved feature table to: {feature_path}")

        return self.feature_df

    # ========================================================================
    # STEP 3: MODEL TRAINING
    # ========================================================================

    def _build_models(self) -> Dict[str, Any]:
        """Build all classifier models."""
        models = {
            "logistic_regression": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=self.random_state,
                        ),
                    ),
                ]
            ),
            "random_forest": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=400,
                            max_depth=None,
                            random_state=self.random_state,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "gradient_boosting": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingClassifier(random_state=self.random_state)),
                ]
            ),
            "hist_gradient_boosting": HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=300,
                max_leaf_nodes=31,
                random_state=self.random_state,
            ),
            "extra_trees": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            random_state=self.random_state,
                            class_weight="balanced",
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        }

        if XGBOOST_AVAILABLE:
            models["xgboost"] = XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=self.random_state,
                n_jobs=-1,
                missing=np.nan,
            )

        return models

    def train_models(self) -> pd.DataFrame:
        """Train all models and compare performance."""
        print("\n[Model Training]")

        # Prepare data
        self.feature_cols = [
            col for col in self.feature_df.columns
            if col not in {"id", "label"}
        ]
        X = self.feature_df[self.feature_cols].copy()
        y = self.feature_df["label"].astype(int)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        print(f"✓ Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        print(f"✓ Features: {self.feature_cols}")

        # Train models
        models = self._build_models()
        results = []

        for name, model in models.items():
            print(f"  Training {name}...", end=" ")
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)

            metrics = {
                "model": name,
                "accuracy": accuracy_score(self.y_test, predictions),
                "precision": precision_score(
                    self.y_test, predictions, zero_division=0
                ),
                "recall": recall_score(self.y_test, predictions, zero_division=0),
                "f1": f1_score(self.y_test, predictions, zero_division=0),
            }

            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(self.X_test)[:, 1]
                    metrics["roc_auc"] = roc_auc_score(self.y_test, probs)
                except Exception:
                    metrics["roc_auc"] = np.nan
            else:
                metrics["roc_auc"] = np.nan

            results.append(metrics)
            self.models[name] = model
            print("✓")

        # Compare and select best
        results_df = (
            pd.DataFrame(results)
            .sort_values(["f1", "accuracy"], ascending=False)
            .reset_index(drop=True)
        )

        self.best_model_name = results_df.iloc[0]["model"]
        self.best_model = self.models[self.best_model_name]

        print(f"\n✓ Best model: {self.best_model_name}")
        print(f"\nModel Comparison:\n{results_df.to_string(index=False)}")

        # Save results
        results_path = self.output_dir / "model_comparison.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Saved comparison to: {results_path}")

        return results_df

    def save_model(self) -> Path:
        """Save best model to disk."""
        if self.best_model is None:
            raise RuntimeError("No trained model. Run train_models() first.")

        model_file = self.model_dir / f"{self.best_model_name}.joblib"
        joblib.dump(self.best_model, model_file)

        print(f"\n✓ Model saved to: {model_file}")

        # Classification report
        final_pred = self.best_model.predict(self.X_test)
        print(f"\nClassification Report:\n{classification_report(self.y_test, final_pred)}")

        return model_file

    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance from best model."""
        if self.best_model is None:
            raise RuntimeError("No trained model. Run train_models() first.")

        # Unwrap pipeline
        clf = self.best_model
        if hasattr(clf, "named_steps"):
            clf = list(clf.named_steps.values())[-1]

        if not hasattr(clf, "feature_importances_"):
            raise ValueError(f"{type(clf)} has no feature_importances_")

        importance_df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": clf.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        importance_df["importance_pct"] = (
            importance_df["importance"] / importance_df["importance"].sum()
        )

        # Save
        importance_path = self.output_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"✓ Feature importance saved to: {importance_path}")
        print(f"\n{importance_df.to_string(index=False)}")

        return importance_df

    # ========================================================================
    # MAIN PIPELINE: TRAIN FROM START TO FINISH
    # ========================================================================

    def train(self, csv_path: str | Path, ragas_evaluator) -> Dict[str, Any]:
        """
        Run complete training pipeline from data to model.

        Args:
            csv_path: Path to labeled CSV
            ragas_evaluator: RAGASEvaluator instance

        Returns:
            Dictionary with results
        """
        print("=" * 80)
        print("RAGAS FILTER - COMPLETE TRAINING PIPELINE")
        print("=" * 80)

        # Load data
        self.load_data(csv_path)
        self.extract_contexts()

        # Compute RAGAS
        self.compute_ragas_features(ragas_evaluator)

        # Train models
        comparison_df = self.train_models()

        # Save model
        self.save_model()

        # Feature importance
        importance_df = self.get_feature_importance()

        print("\n" + "=" * 80)
        print("✓ TRAINING COMPLETE")
        print("=" * 80)

        return {
            "feature_df": self.feature_df,
            "model_comparison": comparison_df,
            "feature_importance": importance_df,
            "best_model": self.best_model_name,
            "model_path": self.model_dir / f"{self.best_model_name}.joblib",
        }
