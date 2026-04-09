"""
Abstract base class for all filtering strategies.

Every filter in the pipeline inherits from BaseFilter, ensuring a
consistent interface: (documents, query) -> scored/filtered documents.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScoredDocument:
    """A document with an associated relevance score."""

    text: str
    score: float
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FilterResult:
    """Outcome of a single filtering operation."""

    kept: List[ScoredDocument]
    removed: List[ScoredDocument]
    filter_name: str
    threshold: float
    reasoning: str = ""


class BaseFilter(ABC):
    """
    Abstract interface that all filtering strategies must implement.

    Subclasses must override ``filter`` and ``score_documents``.
    Thresholds should come from config, never hardcoded.
    """

    def __init__(self, threshold: float, name: Optional[str] = None) -> None:
        self.threshold = threshold
        self.name = name or self.__class__.__name__
        logger.info("Initialised filter %s (threshold=%.2f)", self.name, self.threshold)

    @abstractmethod
    def score_documents(
        self,
        documents: List[str],
        query: str,
    ) -> List[ScoredDocument]:
        """Score each document for relevance to *query*."""

    def filter(
        self,
        documents: List[str],
        query: str,
    ) -> FilterResult:
        """Score documents, then partition into kept / removed."""
        scored = self.score_documents(documents, query)
        kept = [d for d in scored if d.score >= self.threshold]
        removed = [d for d in scored if d.score < self.threshold]

        logger.info(
            "[%s] query=%s  kept=%d  removed=%d  threshold=%.2f",
            self.name,
            query[:60],
            len(kept),
            len(removed),
            self.threshold,
        )

        return FilterResult(
            kept=kept,
            removed=removed,
            filter_name=self.name,
            threshold=self.threshold,
        )

    def get_filtered_texts(
        self,
        documents: List[str],
        query: str,
    ) -> List[str]:
        """Convenience: return only the text of kept documents."""
        result = self.filter(documents, query)
        return [d.text for d in result.kept]
