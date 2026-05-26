"""Config loading tests."""

from __future__ import annotations

from rag_filtering.config.loader import (
    FILTERING_CONFIG,
    load_config_section,
    load_filtering_config,
    repo_root,
    resolve_path,
)


def test_repo_root_exists() -> None:
    root = repo_root()
    assert (root / "configs" / "filtering" / "deberta_filter.yaml").exists()


def test_load_filtering_config() -> None:
    cfg = load_filtering_config(FILTERING_CONFIG)
    assert "learned_filter" in cfg
    assert cfg["learned_filter"]["fp16"] is False


def test_load_learned_filter_section() -> None:
    section = load_config_section(FILTERING_CONFIG, "learned_filter")
    assert section["model_name"]
    assert resolve_path("data/asqa/labeled_asqa.csv").exists()
