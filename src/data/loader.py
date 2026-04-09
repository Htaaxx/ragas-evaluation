"""
Backward-compatible re-export shim.

The original monolithic ``loader.py`` has been split into:
  - base_loader.py     — BaseDataLoader, TrainExample, RetrieverExample
  - hotpotqa_loader.py — HotpotQALoader
  - asqa_loader.py     — ASQALoader

This file re-exports every public name so that existing
``from src.data.loader import X`` statements continue to work.
"""

from .base_loader import BaseDataLoader, RetrieverExample, TrainExample
from .asqa_loader import ASQALoader
from .hotpotqa_loader import HotpotQALoader

__all__ = [
    "BaseDataLoader",
    "TrainExample",
    "RetrieverExample",
    "HotpotQALoader",
    "ASQALoader",
]
