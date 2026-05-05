"""
ASQA (Answer Summaries for Questions which are Ambiguous) dataset loader.

Long-form QA dataset focused on ambiguous factoid questions.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import pandas as pd

from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class ASQALoader(BaseDataLoader):
    """
    Loader for ASQA dataset in CSV format.

    Expected CSV columns:
        id, question, answer, context (JSON string), supporting_facts (JSON string)

    The ``df_valid`` property aliases ``df_dev`` for compatibility with
    code that expects a ``df_valid`` attribute.
    """

    def __init__(self) -> None:
        super().__init__()
        self.df_dev: Optional[pd.DataFrame] = None

    @property  # type: ignore[override]
    def df_valid(self) -> Optional[pd.DataFrame]:  # type: ignore[override]
        """Alias for df_dev to maintain compatibility with base interface."""
        return self.df_dev

    @df_valid.setter
    def df_valid(self, value: Optional[pd.DataFrame]) -> None:
        self.df_dev = value

    def load_data(
        self,
        train_path: str,
        valid_path: str,
        max_train_samples: Optional[int] = None,
        max_dev_samples: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load ASQA data from CSV files."""
        logger.info("Loading ASQA data …")

        self.df_train = self._parse_csv(train_path)
        if max_train_samples:
            self.df_train = self.df_train.head(max_train_samples)

        self.df_dev = self._parse_csv(valid_path)
        if max_dev_samples:
            self.df_dev = self.df_dev.head(max_dev_samples)

        logger.info("Loaded %d training samples", len(self.df_train))
        logger.info("Loaded %d dev samples", len(self.df_dev))

        train_lengths = self.df_train["answer"].str.split().str.len()
        logger.info(
            "Answer length (words): mean=%.1f median=%.1f min=%d max=%d",
            train_lengths.mean(), train_lengths.median(),
            train_lengths.min(), train_lengths.max(),
        )

        return self.df_train, self.df_dev

    @classmethod
    def _parse_csv(cls, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        df["docs"] = df["context"].apply(cls._parse_context)
        return df

    def build_corpus(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[List[str], List[str]]:
        """Build corpus with deduplication by title."""
        df = df if df is not None else self.df_train
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        unique_corpus: dict = {}
        for docs in df["docs"]:
            for doc in docs:
                title = doc["title"]
                if title not in unique_corpus:
                    unique_corpus[title] = doc["text"]

        self.corpus_texts = list(unique_corpus.values())
        self.doc_titles = list(unique_corpus.keys())

        logger.info(
            "Built corpus: %d unique passages from %d questions",
            len(self.corpus_texts), len(df),
        )
        return self.corpus_texts, self.doc_titles

    def get_statistics(self) -> dict:
        stats: dict = {}
        if self.df_train is not None:
            stats["train_samples"] = len(self.df_train)
            stats["train_avg_docs_per_question"] = (
                self.df_train["docs"].apply(len).mean()
            )
            stats["train_avg_answer_length"] = (
                self.df_train["answer"].str.split().str.len().mean()
            )
        if self.df_dev is not None:
            stats["dev_samples"] = len(self.df_dev)
            stats["dev_avg_docs_per_question"] = (
                self.df_dev["docs"].apply(len).mean()
            )
            stats["dev_avg_answer_length"] = (
                self.df_dev["answer"].str.split().str.len().mean()
            )
            stats["valid_samples"] = len(self.df_dev)
            stats["valid_avg_docs_per_question"] = (
                self.df_dev["docs"].apply(len).mean()
            )
        if self.corpus_texts:
            stats["corpus_size"] = len(self.corpus_texts)
            stats["avg_passage_length"] = (
                sum(len(t.split()) for t in self.corpus_texts)
                / len(self.corpus_texts)
            )
        return stats
