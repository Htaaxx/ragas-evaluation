"""
Data loading and preprocessing module.

This module provides utilities for loading and preprocessing the HotpotQA dataset
for RAG training.
"""

from .loader import HotpotQALoader, TrainExample, RetrieverExample

__all__ = [
    "HotpotQALoader",
    "TrainExample",
    "RetrieverExample",
]
