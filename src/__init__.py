"""
RAG Training System

A comprehensive framework for training and fine-tuning RAG models.
"""

from .config import RAGConfig
from .rag_system import RAGSystem

__version__ = "1.0.0"

__all__ = [
    "RAGConfig",
    "RAGSystem",
]
