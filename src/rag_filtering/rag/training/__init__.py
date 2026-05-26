"""
Training modules for RAG components.

This module provides trainers for both retriever and generator models.
"""

from .retriever_trainer import RetrieverTrainer
from .generator_trainer import GeneratorTrainer

__all__ = [
    "RetrieverTrainer",
    "GeneratorTrainer",
]
