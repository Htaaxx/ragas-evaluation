"""
Data loading and preprocessing module.

Provides the ASQA dataset loader and shared data models
(TrainExample, RetrieverExample).
"""

from .base_loader import BaseDataLoader, RetrieverExample, TrainExample
from .asqa_loader import ASQALoader

__all__ = [
    "BaseDataLoader",
    "ASQALoader",
    "TrainExample",
    "RetrieverExample",
]
