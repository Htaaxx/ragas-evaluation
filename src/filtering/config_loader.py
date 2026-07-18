"""YAML config loading from the repository ``configs/`` directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

FILTERING_CONFIG = "configs/filtering/deberta_filter.yaml"


def repo_root() -> Path:
    """Return repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_root() / p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file."""
    full = resolve_path(path)
    if not full.exists():
        raise FileNotFoundError(f"Config not found: {full}")
    with open(full, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def load_config_section(path: str | Path, section: str) -> Dict[str, Any]:
    """Load one top-level section from a YAML config file."""
    return load_yaml(path).get(section, {})


def load_filtering_config(path: str | Path = FILTERING_CONFIG) -> Dict[str, Any]:
    """Load the full filtering config YAML."""
    return load_yaml(path)
