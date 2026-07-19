# Data and splits

## Primary corpus

| File | Role |
|------|------|
| `data/labeled_merged.csv` | Full corpus; train/val exclude rows already in the frozen test file |
| `data/labeled_merged_test.csv` | Frozen holdout used for final DeBERTa / NLI evaluation |

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
test_ratio: 0.2          # only if frozen test CSV is missing
val_ratio: 0.2
seed: 42
reuse_frozen_test: true
```

Default evaluation path (`reuse_frozen_test: true`):

1. Load `data/labeled_merged_test.csv` as the fixed test holdout.
2. Take remaining rows from `data/labeled_merged.csv`.
3. Split remaining base IDs into train / val (`val_ratio`, `seed=42`).

If the frozen test file is missing, fall back to sampling a test split from
base IDs (`test_ratio=0.2`) and writing `labeled_merged_test.csv`.

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
