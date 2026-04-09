"""
Data loading and preprocessing module.

Provides loaders for HotpotQA and ASQA datasets, plus shared
data models (TrainExample, RetrieverExample).
"""

from .base_loader import BaseDataLoader, RetrieverExample, TrainExample
from .asqa_loader import ASQALoader
from .hotpotqa_loader import HotpotQALoader

__all__ = [
    "BaseDataLoader",
    "HotpotQALoader",
    "ASQALoader",
    "TrainExample",
    "RetrieverExample",
]
