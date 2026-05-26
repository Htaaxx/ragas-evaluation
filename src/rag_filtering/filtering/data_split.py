"""
Dataset splitting for the answer quality classifier.

Splits ``labeled_asqa.csv`` into train / validation / test sets by
**base question ID** so that a correct answer (``asqa_X``) and its
hallucinated counterpart (``asqa_Xb``) always land in the same split,
preventing data leakage.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_and_split(
    csv_path: str,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ``labeled_asqa.csv`` and split into train / val / test.

    The split is done at the *base question* level: each unique question
    (identified by stripping the trailing ``b`` from hallucinated IDs) is
    assigned to exactly one split. Both the correct and hallucinated
    versions of a question always stay together.

    Parameters
    ----------
    csv_path:
        Path to ``labeled_asqa.csv``.
    test_ratio:
        Fraction of base question IDs reserved for the test set.
    val_ratio:
        Fraction of the *remaining* base question IDs reserved for
        validation (applied after the test split).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d samples from %s", len(df), csv_path)

    base_ids = df["id"].str.replace(r"b$", "", regex=True).unique()
    logger.info("Found %d unique base question IDs", len(base_ids))

    train_ids, test_ids = train_test_split(
        base_ids, test_size=test_ratio, random_state=seed,
    )
    train_ids, val_ids = train_test_split(
        train_ids, test_size=val_ratio, random_state=seed,
    )

    base_col = df["id"].str.replace(r"b$", "", regex=True)
    train_df = df[base_col.isin(train_ids)].reset_index(drop=True)
    val_df = df[base_col.isin(val_ids)].reset_index(drop=True)
    test_df = df[base_col.isin(test_ids)].reset_index(drop=True)

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_pos = int((split["label"] == 1).sum())
        n_neg = int((split["label"] == 0).sum())
        logger.info("%s: %d samples (pos=%d, neg=%d)", name, len(split), n_pos, n_neg)

    return train_df, val_df, test_df
