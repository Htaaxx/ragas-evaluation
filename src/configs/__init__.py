"""
Configuration loading utilities.

Configs are stored as YAML files in this directory.
Use ``load_config`` to read any config file into a dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

CONFIGS_DIR = Path(__file__).parent


def load_config(name: str) -> Dict[str, Any]:
    """
    Load a YAML config file by name (without extension).

    Args:
        name: Config filename stem, e.g. ``"filtering"`` or ``"pipeline"``.

    Returns:
        Parsed config as a nested dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
