"""Smoke tests for DeBERTa/NLI module imports on main."""

from __future__ import annotations


def test_filtering_public_api() -> None:
    from src.filtering import (
        AnswerQualityClassifier,
        FilterDecision,
        NLIAnswerFilter,
        RagasFilter,
        load_and_split,
        select_threshold_min_fpr,
        to_base_id,
        train_classifier,
    )

    assert callable(load_and_split)
    assert callable(to_base_id)
    assert callable(select_threshold_min_fpr)
    assert callable(train_classifier)
    assert AnswerQualityClassifier is not None
    assert NLIAnswerFilter is not None
    assert RagasFilter is not None
    assert FilterDecision is not None


def test_config_loader() -> None:
    from src.filtering.config_loader import FILTERING_CONFIG, load_yaml, resolve_path

    cfg = load_yaml(FILTERING_CONFIG)
    assert "learned_filter" in cfg
    assert "nli_filter" in cfg
    assert resolve_path("data/labeled_merged.csv").name == "labeled_merged.csv"
