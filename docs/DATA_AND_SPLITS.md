# Data and splits

## Primary corpus

| File | Role |
|------|------|
| `data/labeled_merged.csv` | Full labeled train/val/test source (~9.8k rows) |
| `data/labeled_merged_test.csv` | Frozen holdout written by the leakage-safe split |

Merged composition (approx.): ASQA + MS MARCO + WikiEval. Labels are balanced correct / hallucinated.

### Schema

Required columns:

| Column | Meaning |
|--------|---------|
| `id` | Unique sample id |
| `question` | Question text |
| `context` | Retrieved / supporting passage (plain text on merged) |
| `answer` | Answer to judge for faithfulness |
| `label` | `1` = correct / faithful, `0` = hallucinated |
| `dataset` | Source tag: `asqa`, `msmarco`, `wikieval`, … |

### ID pairing (`_hallu`)

On `main`, hallucinated twins use a `_hallu` suffix:

- correct: `asqa_0`
- hallucinated: `asqa_0_hallu`

Legacy ASQA used a trailing `b` (`asqa_0b`). Split code accepts both via `to_base_id()` in `src/filtering/data_split.py`.

## Leakage-safe split

Configured in `configs/experiments/filter_training.yaml`:

```yaml
test_ratio: 0.2
val_ratio: 0.2   # of remaining after test split
seed: 42
```

Algorithm (`load_and_split`):

1. Map every `id` → base id (`strip _hallu`, then trailing `b`).
2. `train_test_split(base_ids, test_size=0.2, random_state=42)`.
3. Split remaining base ids into train / val with `val_ratio=0.2`.
4. Assign all rows whose base id falls in a fold to that fold (pairs stay together).
5. Optionally persist test rows to `data/labeled_merged_test.csv`.

This matches the intent of a simple `train_test_split(df, test_size=0.2, random_state=42)` while preventing correct/hallu twins from crossing the train–test boundary.

## Other data folders

| Path | Notes |
|------|-------|
| `data/asqa/` | ASQA-only labeled / raw CSVs |
| `data/ms-marco/` | MS MARCO labeled / hallu |
| `data/wikiEval/` | WikiEval labeled / raw |

Prefer `labeled_merged.csv` for thesis-wide DeBERTa / RAGAS comparisons so all methods share the same corpus.

## Reproducibility checklist

- Always note which CSV + seed + ratios produced a result JSON.
- After changing split logic, regenerate `labeled_merged_test.csv` and retrain; do not mix old checkpoints with a new holdout.
- Classification reports break metrics down by `dataset` plus an `Overall` row (see [EVALUATION.md](EVALUATION.md)).
