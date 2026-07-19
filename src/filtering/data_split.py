"""
Dataset splitting for the answer quality classifier.

Splits a labeled CSV into train / validation / test by **base question ID**
so that a correct answer and its hallucinated counterpart always land in
the same split, preventing data leakage.

Supports both ID schemes:
- ``asqa_X`` / ``asqa_X_hallu`` (merged thesis corpus on ``main``)
- ``asqa_X`` / ``asqa_Xb`` (legacy ASQA labeling)

When a frozen ``test_csv_path`` already exists, that file is used as the
held-out test set and train/val are carved from the remaining rows of the
full labeled CSV (no re-sampling of the test holdout).
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
    reuse_frozen_test: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a labeled CSV and split into train / val / test by base ID.

    Parameters
    ----------
    csv_path:
        Path to the full labeled dataset (e.g. ``data/labeled_merged.csv``).
    test_ratio:
        Fraction of base question IDs reserved for the test set when a
        frozen test CSV is not reused.
    val_ratio:
        Fraction of the *remaining* base question IDs reserved for
        validation (applied after the test split / freeze exclusion).
    seed:
        Random seed for reproducibility.
    test_csv_path:
        Frozen holdout path (e.g. ``data/labeled_merged_test.csv``).
        If the file exists and ``reuse_frozen_test`` is True, it is loaded
        as ``test_df`` and train/val are taken from the rest of ``csv_path``.
        If missing, a new test split is created and written here when set.
    reuse_frozen_test:
        Prefer an existing ``test_csv_path`` instead of resampling test.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d samples from %s", len(df), csv_path)

    frozen_path = Path(test_csv_path) if test_csv_path is not None else None
    if (
        reuse_frozen_test
        and frozen_path is not None
        and frozen_path.is_file()
    ):
        test_df = pd.read_csv(frozen_path).reset_index(drop=True)
        test_ids = set(test_df["id"].astype(str))
        remain_df = df[~df["id"].astype(str).isin(test_ids)].reset_index(drop=True)
        logger.info(
            "Using frozen test set %s (%d rows); remaining for train/val: %d",
            frozen_path,
            len(test_df),
            len(remain_df),
        )

        remain_base = remain_df["id"].map(to_base_id)
        base_ids = remain_base.unique()
        train_ids, val_ids = train_test_split(
            base_ids, test_size=val_ratio, random_state=seed,
        )
        train_df = remain_df[remain_base.isin(train_ids)].reset_index(drop=True)
        val_df = remain_df[remain_base.isin(val_ids)].reset_index(drop=True)
    else:
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

        if frozen_path is not None:
            frozen_path.parent.mkdir(parents=True, exist_ok=True)
            test_df.to_csv(frozen_path, index=False)
            logger.info("Saved test split to %s (%d rows)", frozen_path, len(test_df))

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_pos = int((split["label"] == 1).sum())
        n_neg = int((split["label"] == 0).sum())
        logger.info("%s: %d samples (pos=%d, neg=%d)", name, len(split), n_pos, n_neg)

    # Guard: no row id overlap between train/val and frozen test
    train_val_ids = set(train_df["id"].astype(str)) | set(val_df["id"].astype(str))
    overlap = train_val_ids & set(test_df["id"].astype(str))
    if overlap:
        raise ValueError(
            f"Train/val overlap with test holdout ({len(overlap)} ids). "
            "Regenerate or fix labeled_merged_test.csv."
        )

    return train_df, val_df, test_df
