"""
ragas_feature_extractor.py

Dataset-agnostic RAGAS feature extraction utilities.

Task:
    Given question, context, answer:
    compute RAGAS metrics used as features for contextual faithfulness filtering.

Default RAGAS features:
    - faithfulness
    - answer_relevancy
    - context_precision
    - context_recall
    - context_relevancy
    - answer_correctness
    - answer_similarity
    
Required:
    - id
    - question
    - answer
    - context

Optional:
    - label
    - gold_ans / gold_answer / reference / reference_answer

Context handling:
    The context column should already be processed, but this module supports:
        - list[str]
        - stringified list[str]
        - JSON list
        - plain string


Notes:
    This module only creates features.
    Use RagasFilterTrainer for training and RagasFilter for inference/evaluation.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

from .helper import (
    DEFAULT_RAGAS_FEATURE_COLS,
    RAW_REQUIRED_COLS,
    _ensure_path,
    _normalize_col_aliases,
    parse_context,
)


class RagasFeatureExtractor:
    """
    Raw data -> RAGAS features.

    Reusable for both training and inference.
    `ragas_evaluator` is usually your existing src.filtering.ragas.RAGAS wrapper.
    """

    def __init__(
        self,
        ragas_evaluator: Any,
        feature_cols: Optional[Sequence[str]] = None,
        include_raw_columns: bool = False,
        id_col: str = "id",
        question_col: str = "question",
        answer_col: str = "answer",
        context_col: str = "context",
    ) -> None:
        self.ragas_evaluator = ragas_evaluator
        self.feature_cols = list(feature_cols or DEFAULT_RAGAS_FEATURE_COLS)
        self.include_raw_columns = include_raw_columns
        self.id_col = id_col
        self.question_col = question_col
        self.answer_col = answer_col
        self.context_col = context_col

    def load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        data_path = _ensure_path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        suffix = data_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(data_path)
        elif suffix == ".jsonl":
            df = pd.read_json(data_path, lines=True)
        elif suffix == ".json":
            df = pd.read_json(data_path)
        elif suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported data file type: {suffix}")
        return _normalize_col_aliases(df)

    def validate_schema(self, df: pd.DataFrame) -> None:
        missing = [c for c in RAW_REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Expected at least {RAW_REQUIRED_COLS}. Current columns: {list(df.columns)}"
            )

    def prepare_data(self, data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, (str, Path)):
            df = self.load_data(data)
        else:
            df = _normalize_col_aliases(data.copy())
        self.validate_schema(df)
        df["_parsed_context"] = df[self.context_col].apply(parse_context)
        return df

    def transform(
        self,
        data: Union[str, Path, pd.DataFrame],
        feature_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        batch_size: int = 50,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        df = self.prepare_data(data)

        questions = df[self.question_col].fillna("").astype(str).tolist()
        answers = df[self.answer_col].fillna("").astype(str).tolist()
        contexts = df["_parsed_context"].tolist()

        if checkpoint_path is not None and hasattr(self.ragas_evaluator, "evaluate_checkpoint"):
            ragas_result = self.ragas_evaluator.evaluate_checkpoint(
                questions=questions,
                answers=answers,
                contexts=contexts,
                batch_size=batch_size,
                save_path=checkpoint_path,
                show_progress=show_progress,
            )
        else:
            ragas_result = self.ragas_evaluator.evaluate(
                questions=questions,
                answers=answers,
                contexts=contexts,
                show_progress=show_progress,
            )

        if hasattr(ragas_result, "to_pandas"):
            ragas_df = ragas_result.to_pandas()
        elif isinstance(ragas_result, pd.DataFrame):
            ragas_df = ragas_result.copy()
        else:
            ragas_df = pd.DataFrame(ragas_result)

        kept_features = [c for c in self.feature_cols if c in ragas_df.columns]
        if not kept_features:
            kept_features = ragas_df.select_dtypes(include=[np.number]).columns.tolist()
            logger.warning("No requested RAGAS cols found. Falling back to numeric cols: %s", kept_features)
        if not kept_features:
            raise ValueError("RAGAS result does not contain usable numeric feature columns.")

        feature_df = ragas_df[kept_features].reset_index(drop=True)

        if self.include_raw_columns:
            raw_cols = [c for c in df.columns if c != "_parsed_context"]
            out = pd.concat([df[raw_cols].reset_index(drop=True), feature_df], axis=1)
            out["parsed_context"] = df["_parsed_context"].apply(lambda x: json.dumps(x, ensure_ascii=False))
        else:
            out = pd.concat([df[[self.id_col]].reset_index(drop=True), feature_df], axis=1)
            # if raw df have label column, keep it for training filter model
            if "label" in df.columns:
                out["label"] = df["label"]
            if "dataset" in df.columns:
                out["dataset"] = df["dataset"]

        if feature_path is not None:
            feature_path = _ensure_path(feature_path)
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(feature_path, index=False, encoding="utf-8-sig")

        return out

    def transform_from_path(
        self,
        data_path: Union[str, Path],
        feature_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        batch_size: int = 50,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        return self.transform(
            data=data_path,
            feature_path=feature_path,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            show_progress=show_progress,
        )

def build_ragas_features(
    data_path: Union[str, Path],
    ragas_evaluator: Any,
    feature_path: Optional[Union[str, Path]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    batch_size: int = 50,
) -> pd.DataFrame:
    extractor = RagasFeatureExtractor(ragas_evaluator=ragas_evaluator)
    return extractor.transform_from_path(
        data_path=data_path,
        feature_path=feature_path,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
    )
