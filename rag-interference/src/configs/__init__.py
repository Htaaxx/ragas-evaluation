"""Configuration loading for the rag-interference verifier experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

CONFIGS_DIR = Path(__file__).parent
REPO_ROOT = CONFIGS_DIR.parent.parent.parent


def load_config(name: str = "rag_verifier") -> Dict[str, Any]:
    """Load a YAML config from ``rag-interference/src/configs/``."""
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_repo_path(path_str: str) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path
