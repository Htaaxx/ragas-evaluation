"""
HotpotQA dataset loader.

Loads the multi-hop QA dataset from CSV files produced by the
data_collection notebook.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import pandas as pd

from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class HotpotQALoader(BaseDataLoader):
    """
    Loader for HotpotQA dataset in CSV format.

    Expected CSV columns:
        question, answer, context (JSON string), supporting_facts (JSON string)
    """

    def load_data(
        self,
        train_path: str,
        valid_path: str,
        max_train_samples: Optional[int] = None,
        max_valid_samples: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load HotpotQA data from CSV files."""
        logger.info("Loading HotpotQA data …")

        self.df_train = self._parse_csv(train_path)
        if max_train_samples:
            self.df_train = self.df_train.head(max_train_samples)

        self.df_valid = self._parse_csv(valid_path)
        if max_valid_samples:
            self.df_valid = self.df_valid.head(max_valid_samples)

        logger.info("Loaded %d training samples", len(self.df_train))
        logger.info("Loaded %d validation samples", len(self.df_valid))

        return self.df_train, self.df_valid

    @classmethod
    def _parse_csv(cls, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        df["docs"] = df["context"].apply(cls._parse_context)
        return df

    def get_statistics(self) -> dict:
        stats: dict = {}
        if self.df_train is not None:
            stats["train_samples"] = len(self.df_train)
            stats["train_avg_docs_per_question"] = (
                self.df_train["docs"].apply(len).mean()
            )
        if self.df_valid is not None:
            stats["valid_samples"] = len(self.df_valid)
            stats["valid_avg_docs_per_question"] = (
                self.df_valid["docs"].apply(len).mean()
            )
        if self.corpus_texts:
            stats["corpus_size"] = len(self.corpus_texts)
            stats["avg_passage_length"] = (
                sum(len(t.split()) for t in self.corpus_texts)
                / len(self.corpus_texts)
            )
        return stats
