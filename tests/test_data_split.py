"""Data split leakage tests."""

from __future__ import annotations

from rag_filtering.filtering.data_split import load_and_split


def test_paired_ids_stay_in_same_split(tmp_path) -> None:
    csv = tmp_path / "mini.csv"
    rows = ["id,question,answer,context,label"]
    for i in range(1, 5):
        rows.append(f"asqa_{i},q,a{i},c,1")
        rows.append(f"asqa_{i}b,q,a{i}b,c,0")
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    train_df, val_df, test_df = load_and_split(
        str(csv), test_ratio=0.25, val_ratio=0.33, seed=42,
    )
    all_ids = set(train_df["id"]) | set(val_df["id"]) | set(test_df["id"])

    for base in ["asqa_1", "asqa_2", "asqa_3", "asqa_4"]:
        in_train = f"{base}" in train_df["id"].values or f"{base}b" in train_df["id"].values
        in_val = f"{base}" in val_df["id"].values or f"{base}b" in val_df["id"].values
        in_test = f"{base}" in test_df["id"].values or f"{base}b" in test_df["id"].values
        assert sum([in_train, in_val, in_test]) == 1, f"{base} pair leaked across splits"
    assert len(all_ids) == 8
