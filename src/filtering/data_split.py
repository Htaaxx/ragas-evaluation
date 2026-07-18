"""
Dataset splitting for the answer quality classifier.

Splits a labeled CSV into train / validation / test by **base question ID**
so that a correct answer and its hallucinated counterpart always land in
the same split, preventing data leakage.

Supports both ID schemes:
- ``asqa_X`` / ``asqa_X_hallu`` (merged thesis corpus on ``main``)
- ``asqa_X`` / ``asqa_Xb`` (legacy ASQA labeling)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

_HALLU_SUFFIX_RE = re.compile(r"_hallu$", flags=re.IGNORECASE)
_LEGACY_B_SUFFIX_RE = re.compile(r"b$")


def to_base_id(sample_id: str) -> str:
    """Map a sample id to its leakage-safe base question id."""
    text = str(sample_id)
    text = _HALLU_SUFFIX_RE.sub("", text)
    text = _LEGACY_B_SUFFIX_RE.sub("", text)
    return text


def load_and_split(
    csv_path: str,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    seed: int = 42,
    test_csv_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a labeled CSV and split into train / val / test by base ID.

    Parameters
    ----------
    csv_path:
        Path to the labeled dataset (e.g. ``data/labeled_merged.csv``).
    test_ratio:
        Fraction of base question IDs reserved for the test set.
    val_ratio:
        Fraction of the *remaining* base question IDs reserved for
        validation (applied after the test split).
    seed:
        Random seed for reproducibility.
    test_csv_path:
        If set, persist the held-out test split to this CSV path.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d samples from %s", len(df), csv_path)

    base_col = df["id"].map(to_base_id)
    base_ids = base_col.unique()
    logger.info("Found %d unique base question IDs", len(base_ids))

    train_ids, test_ids = train_test_split(
        base_ids, test_size=test_ratio, random_state=seed,
    )
    train_ids, val_ids = train_test_split(
        train_ids, test_size=val_ratio, random_state=seed,
    )

    train_df = df[base_col.isin(train_ids)].reset_index(drop=True)
    val_df = df[base_col.isin(val_ids)].reset_index(drop=True)
    test_df = df[base_col.isin(test_ids)].reset_index(drop=True)

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_pos = int((split["label"] == 1).sum())
        n_neg = int((split["label"] == 0).sum())
        logger.info("%s: %d samples (pos=%d, neg=%d)", name, len(split), n_pos, n_neg)

    if test_csv_path is not None:
        out = Path(test_csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(out, index=False)
        logger.info("Saved test split to %s (%d rows)", out, len(test_df))

    return train_df, val_df, test_df
