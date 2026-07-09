"""Smoke tests for the RAGAS-feature filter (no RAGAS/OpenAI calls)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rag_filtering.filtering import (
    RagasFilter,
    RagasFilterTrainer,
)
from rag_filtering.filtering.helper import parse_context


def test_public_api_imports() -> None:
    from rag_filtering.filtering import (  # noqa: F401
        RagasFeatureExtractor,
        build_ragas_features,
        run_ragas_filter,
        train_ragas_filter,
    )
    from rag_filtering.evaluation import RAGAS  # noqa: F401


def test_parse_context_variants() -> None:
    assert parse_context(["a", "b"]) == ["a", "b"]
    assert parse_context('["x", "y"]') == ["x", "y"]
    assert parse_context("- (title) plain text") == ["- (title) plain text"]
    assert parse_context(None) == []


def _synthetic_features(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    half = n // 2
    labels = np.array([1] * half + [0] * half)
    faithfulness = np.concatenate(
        [rng.uniform(0.6, 1.0, half), rng.uniform(0.0, 0.4, half)]
    )
    answer_relevancy = np.concatenate(
        [rng.uniform(0.5, 1.0, half), rng.uniform(0.0, 0.5, half)]
    )
    return pd.DataFrame(
        {
            "id": [f"s_{i}" for i in range(n)],
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "label": labels,
        }
    )


def test_train_and_predict_roundtrip(tmp_path) -> None:
    df = _synthetic_features()

    trainer = RagasFilterTrainer(
        feature_data=df,
        output_dir=str(tmp_path / "model"),
        feature_cols=["faithfulness", "answer_relevancy"],
        test_size=0.3,
        random_state=42,
    )
    out = trainer.run()
    model_path = out["model_path"]
    assert model_path.exists()
    assert out["best_model_name"]

    ragas_filter = RagasFilter(
        model_path=model_path,
        output_dir=str(tmp_path / "results"),
        threshold=0.5,
    )
    predicted = ragas_filter.predict(feature_df=df)
    assert "filter_label" in predicted.columns
    assert "filter_confidence" in predicted.columns
    assert set(predicted["filter_label"].unique()).issubset({0, 1})

    # Threshold selection on labeled data should return a usable threshold.
    result = ragas_filter.select_threshold(predicted, min_recall=0.7)
    assert 0.05 <= result["threshold"] <= 0.99

    # Evaluation against labels produces the thesis FilterResult metrics.
    evaluation = ragas_filter.evaluate(predicted)
    assert "filter_result" in evaluation
    assert "fp" in evaluation["filter_result"]
