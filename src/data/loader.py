"""
Backward-compatible re-export shim.

The original monolithic ``loader.py`` has been split into:
  - base_loader.py  — BaseDataLoader, TrainExample, RetrieverExample
  - asqa_loader.py  — ASQALoader

This file re-exports every public name so that existing
``from src.data.loader import X`` statements continue to work.
"""

from .base_loader import BaseDataLoader, RetrieverExample, TrainExample
from .asqa_loader import ASQALoader

__all__ = [
    "BaseDataLoader",
    "TrainExample",
    "RetrieverExample",
    "ASQALoader",
]
