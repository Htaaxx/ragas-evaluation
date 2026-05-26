"""
Abstract base loader and shared data models for dataset loading.

Concrete loaders (e.g. ASQALoader) inherit from BaseDataLoader
to ensure a consistent interface across datasets.
"""

from __future__ import annotations

import ast
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainExample:
    """Training example for generator."""

    question: str
    answer: str
    contexts: Optional[Sequence[str]] = None


@dataclass
class RetrieverExample:
    """Training example for retriever."""

    question: str
    positive_passage: str
    negative_passages: Optional[Sequence[str]] = None


class BaseDataLoader(ABC):
    """
    Abstract base for dataset loaders.

    Subclasses must implement ``load_data`` and may override
    ``build_corpus``, ``create_retriever_examples``, etc.
    """

    def __init__(self) -> None:
        self.df_train: Optional[pd.DataFrame] = None
        self.df_valid: Optional[pd.DataFrame] = None
        self.corpus_texts: List[str] = []
        self.doc_titles: List[str] = []

    @abstractmethod
    def load_data(self, train_path: str, valid_path: str, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and parse dataset into train/valid DataFrames."""

    @staticmethod
    def _parse_context(ctx_str: str) -> List[dict]:
        """Parse context JSON string into list of {title, text} dicts."""
        try:
            ctx = ast.literal_eval(ctx_str)
            docs = []
            if isinstance(ctx, dict):
                titles = ctx.get("title", [])
                sentences_list = ctx.get("sentences", [])
            elif isinstance(ctx, list):
                titles = [item[0] for item in ctx]
                sentences_list = [item[1] for item in ctx]
            else:
                return []

            for title, sentences in zip(titles, sentences_list):
                if sentences and len(sentences) > 0:
                    text = " ".join(sentences)
                    docs.append({"title": title, "text": text})
            return docs
        except Exception as exc:
            logger.warning("Error parsing context: %s", exc)
            return []

    @staticmethod
    def _parse_supporting_facts(raw: object) -> Optional[List[str]]:
        """Extract positive document titles from supporting_facts field."""
        try:
            if isinstance(raw, str):
                raw = ast.literal_eval(raw)
            if isinstance(raw, dict):
                return raw.get("title", [])
            if isinstance(raw, list):
                return list({item[0] for item in raw})
        except Exception:
            pass
        return None

    def build_corpus(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[List[str], List[str]]:
        """Build corpus by flattening all documents in the DataFrame."""
        df = df if df is not None else self.df_train
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        corpus: List[dict] = []
        for docs in df["docs"]:
            corpus.extend(docs)

        self.corpus_texts = [doc["text"] for doc in corpus]
        self.doc_titles = [doc["title"] for doc in corpus]

        logger.info(
            "Built corpus: %d passages from %d questions",
            len(self.corpus_texts), len(df),
        )
        return self.corpus_texts, self.doc_titles

    def create_retriever_examples(
        self,
        df: Optional[pd.DataFrame] = None,
        max_examples: Optional[int] = None,
    ) -> List[RetrieverExample]:
        """Create contrastive training examples for the retriever."""
        df = df if df is not None else self.df_train
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        examples: List[RetrieverExample] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Retriever examples"):
            if max_examples and len(examples) >= max_examples:
                break

            pos_titles = self._parse_supporting_facts(row["supporting_facts"])
            if pos_titles is None:
                continue

            docs = row["docs"]
            pos_docs = [d["text"] for d in docs if d["title"] in pos_titles]
            neg_docs = [d["text"] for d in docs if d["title"] not in pos_titles]

            if not pos_docs or not neg_docs:
                continue

            examples.append(
                RetrieverExample(
                    question=row["question"],
                    positive_passage=random.choice(pos_docs),
                    negative_passages=neg_docs if neg_docs else None,
                )
            )

        logger.info("Created %d retriever training examples", len(examples))
        return examples

    def create_generator_examples(
        self,
        df: Optional[pd.DataFrame] = None,
        max_examples: Optional[int] = None,
        include_contexts: bool = False,
    ) -> List[TrainExample]:
        """Create training examples for the generator."""
        df = df if df is not None else self.df_train
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        examples: List[TrainExample] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generator examples"):
            if max_examples and len(examples) >= max_examples:
                break

            contexts = None
            if include_contexts:
                contexts = [d["text"] for d in row["docs"]]

            examples.append(
                TrainExample(
                    question=row["question"],
                    answer=row["answer"],
                    contexts=contexts,
                )
            )

        logger.info("Created %d generator training examples", len(examples))
        return examples

    @abstractmethod
    def get_statistics(self) -> dict:
        """Return dataset statistics."""
