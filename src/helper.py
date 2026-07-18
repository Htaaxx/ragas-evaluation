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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


DEFAULT_RAGAS_FEATURE_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "context_relevancy",
    "answer_correctness",
    "answer_similarity",
]

RAW_REQUIRED_COLS = ["id", "question", "answer", "context"]

GOLD_ANSWER_CANDIDATES = [
    "gold_ans",
    "gold_answer",
    "reference",
    "reference_answer",
    "ground_truth",
]


def _ensure_path(path: Union[str, Path]) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _safe_json_dump(obj: Any, path: Union[str, Path]) -> None:
    path = _ensure_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _normalize_col_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column aliases into canonical schema."""
    df = df.copy()
    alias_map = {
        "qid": "id",
        "sample_id": "id",
        "query": "question",
        "prompt": "question",
        "prediction": "answer",
        "predicted_answer": "answer",
        "response": "answer",
        "generated_answer": "answer",
        "contexts": "context",
        "retrieved_context": "context",
        "retrieved_contexts": "context",
        "gold_answer": "gold_ans",
        "reference": "gold_ans",
        "reference_answer": "gold_ans",
        "ground_truth": "gold_ans",
        "target": "label",
        "y": "label",
    }
    rename = {}
    for old, new in alias_map.items():
        if old in df.columns and new not in df.columns:
            rename[old] = new
    df = df.rename(columns=rename) if rename else df
    #df['id'] = df['id'].astype(str)
    return df


def _context_item_to_text(item: Any) -> str:
    if item is None:
        return ""
    try:
        if isinstance(item, float) and np.isnan(item):
            return ""
    except Exception:
        pass
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in [
            "text", "content", "passage_text", "passage", "context",
            "body", "sentence", "paragraph",
        ]:
            if key in item and item[key] is not None:
                return str(item[key]).strip()
        values = []
        for value in item.values():
            if isinstance(value, (str, int, float)):
                values.append(str(value))
        return " ".join(values).strip()
    return str(item).strip()


def parse_context(context_raw: Any) -> List[str]:
    """
    Robust context parser.

    Accepts list[str], stringified list, JSON list, dicts, or plain string.
    Returns list[str].
    """
    if context_raw is None:
        return []
    try:
        if isinstance(context_raw, float) and np.isnan(context_raw):
            return []
    except Exception:
        pass

    if isinstance(context_raw, np.ndarray):
        context_raw = context_raw.tolist()
    if isinstance(context_raw, tuple):
        context_raw = list(context_raw)
    if isinstance(context_raw, list):
        out = []
        for item in context_raw:
            text = _context_item_to_text(item)
            if text:
                out.append(text)
        return out
    if isinstance(context_raw, dict):
        text = _context_item_to_text(context_raw)
        return [text] if text else []
    if isinstance(context_raw, str):
        s = context_raw.strip()
        if not s:
            return []
        try:
            return parse_context(json.loads(s))
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, dict)):
                return parse_context(parsed)
        except Exception:
            pass
        return [s]

    text = str(context_raw).strip()
    return [text] if text else []


def _get_gold_col(df: pd.DataFrame) -> Optional[str]:
    for col in GOLD_ANSWER_CANDIDATES:
        if col in df.columns:
            return col
    return None


def get_default_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    models: Dict[str, BaseEstimator] = {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                )),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    random_state=random_state,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                )),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=300,
            max_leaf_nodes=31,
            random_state=random_state,
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(
                    n_estimators=500,
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                )),
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
            random_state=random_state,
            n_jobs=-1,
            missing=np.nan,
        )
    return models