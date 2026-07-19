"""Tests for leakage-safe base-ID splitting (_hallu + legacy b suffix)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.filtering.data_split import load_and_split, to_base_id


def test_to_base_id_hallu_and_legacy() -> None:
    assert to_base_id("asqa_0") == "asqa_0"
    assert to_base_id("asqa_0_hallu") == "asqa_0"
    assert to_base_id("asqa_0b") == "asqa_0"
    assert to_base_id("14_pos") == "14_pos"


def test_pairs_stay_together(tmp_path: Path) -> None:
    rows = []
    for i in range(20):
        rows.append(
            {
                "id": f"asqa_{i}",
                "question": f"q{i}",
                "context": f"c{i}",
                "answer": f"a{i}",
                "label": 1,
                "dataset": "asqa",
            }
        )
        rows.append(
            {
                "id": f"asqa_{i}_hallu",
                "question": f"q{i}",
                "context": f"c{i}",
                "answer": f"h{i}",
                "label": 0,
                "dataset": "asqa",
            }
        )
    csv_path = tmp_path / "labeled.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    test_csv = tmp_path / "test.csv"

    train_df, val_df, test_df = load_and_split(
        str(csv_path),
        test_ratio=0.2,
        val_ratio=0.2,
        seed=42,
        test_csv_path=test_csv,
    )

    assert len(train_df) + len(val_df) + len(test_df) == 40
    assert test_csv.exists()

    for split in (train_df, val_df, test_df):
        bases = set(split["id"].map(to_base_id))
        for base in bases:
            group = split[split["id"].map(to_base_id) == base]
            assert set(group["label"]) == {0, 1}, f"pair split across folds for {base}"


def test_load_and_split_merged_smoke() -> None:
    csv_path = Path("data/labeled_merged.csv")
    if not csv_path.exists():
        return
    train_df, val_df, test_df = load_and_split(
        str(csv_path),
        test_ratio=0.2,
        val_ratio=0.2,
        seed=42,
        reuse_frozen_test=False,
    )
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(pd.read_csv(csv_path))
    # ~20% test by base IDs; allow some slack for uneven pair sizes
    assert 0.15 * total <= len(test_df) <= 0.30 * total


def test_reuse_frozen_test_csv(tmp_path: Path) -> None:
    rows = []
    for i in range(20):
        rows.append(
            {
                "id": f"asqa_{i}",
                "question": f"q{i}",
                "context": f"c{i}",
                "answer": f"a{i}",
                "label": 1,
                "dataset": "asqa",
            }
        )
        rows.append(
            {
                "id": f"asqa_{i}_hallu",
                "question": f"q{i}",
                "context": f"c{i}",
                "answer": f"h{i}",
                "label": 0,
                "dataset": "asqa",
            }
        )
    full = tmp_path / "labeled.csv"
    frozen = tmp_path / "labeled_test.csv"
    pd.DataFrame(rows).to_csv(full, index=False)
    # Freeze first 4 base ids (8 rows) as test
    test_rows = [r for r in rows if to_base_id(r["id"]) in {"asqa_0", "asqa_1", "asqa_2", "asqa_3"}]
    pd.DataFrame(test_rows).to_csv(frozen, index=False)

    train_df, val_df, test_df = load_and_split(
        str(full),
        val_ratio=0.25,
        seed=42,
        test_csv_path=frozen,
        reuse_frozen_test=True,
    )
    assert len(test_df) == 8
    assert set(test_df["id"].astype(str)).isdisjoint(set(train_df["id"].astype(str)))
    assert set(test_df["id"].astype(str)).isdisjoint(set(val_df["id"].astype(str)))
    assert len(train_df) + len(val_df) + len(test_df) == 40
