# DeBERTa / NLI baseline

Thesis **baseline** faithfulness filter: fine-tune DeBERTa as a binary classifier (context entails answer?) and compare to zero-shot NLI.

Model: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`  
Notebook: `notebooks/5_deberta_nli_baseline.ipynb`  
Script: `scripts/run_deberta_nli_baseline.py`  
Config: `configs/experiments/filter_training.yaml` + `configs/filtering/deberta_filter.yaml`

Related: [DATA_AND_SPLITS.md](DATA_AND_SPLITS.md), [EVALUATION.md](EVALUATION.md).

## Protocol (locked)

1. Load `data/labeled_merged.csv`.
2. Leakage-safe base-ID split: `test_size=0.2`, `random_state=42`; freeze `data/labeled_merged_test.csv`; carve val from remaining train.
3. Pre-training gates (notebook or script):
   - Label mapping check (`1` = correct, `0` = hallu)
   - Spot-check paired samples
   - Truncation collision diagnostic (< 5%)
   - `overfit_sanity_check` (train F1 ≥ 0.95) — **hard gate**
4. Train DeBERTa **3 identical runs** (same split/seed).
5. Per run: select threshold on val with min-FPR @ recall ≥ 0.70; evaluate on frozen test.
6. Run zero-shot NLI once on the same test set (val-thresholded).
7. Aggregate mean±std; write classification reports.

## Local / Kaggle: what to run

### Option A — Notebook (good on Kaggle)

1. Put the repo on the machine so `notebooks/` sits next to `src/`, `configs/`, `data/`.
2. Open `notebooks/5_deberta_nli_baseline.ipynb`.
3. Run setup → split → gate cells.
4. Set `RUN_TRAINING = True` and run the training cell.

That cell calls:

```bash
python scripts/run_deberta_nli_baseline.py \
  --config configs/experiments/filter_training.yaml \
  --skip-overfit-gate
```

(`--skip-overfit-gate` only because the notebook already ran the gate.)

### Option B — Headless script

From repo root (GPU session):

```bash
pip install -r requirements.txt
# CUDA torch if needed, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu124

python scripts/run_deberta_nli_baseline.py \
  --config configs/experiments/filter_training.yaml
```

Resume eval only (checkpoints already present):

```bash
python scripts/run_deberta_nli_baseline.py --skip-train --skip-overfit-gate
```

### Kaggle tips

- Use a GPU runtime (T4+). On 16 GB GPUs you can set `batch_size: 4` in `deberta_filter.yaml`; 4 GB laptops stay at `1`.
- Keep `fp16: false`.
- If HuggingFace download returns **401**, clear an expired token (`hf auth logout` / delete cached HF token) before retrying.
- Save / download artifacts before the session ends (see below). One session should cover all 3 runs + NLI.

## Artifacts

| Path | Contents |
|------|----------|
| `models/answer_filter/run_{1,2,3}/` | Checkpoints |
| `results/deberta_nli/run_{k}/threshold_selection.json` | Val threshold sweep |
| `results/deberta_nli/run_{k}/learned_filter_test_results.json` | Test FilterResult |
| `results/deberta_nli/run_{k}/test_predictions.csv` | Per-row preds |
| `results/deberta_nli/run_{k}/classification_report.csv` | Per-dataset table |
| `results/deberta_nli/summary_classification_report.csv` | **Mean over 3 runs** (thesis table) |
| `results/deberta_nli/summary.json` | Full aggregate + comparison |
| `results/deberta_nli/nli_zeroshot/` | Zero-shot NLI results |
| `results/deberta_nli/no_filter_classification_report.csv` | Accept-all baseline |

Primary table for the thesis write-up:

`results/deberta_nli/summary_classification_report.csv`

Columns: `dataset,num_samples,accepted,acceptance_rate,accuracy,precision,recall,f1,roc_auc`.

## Code map

| Module | Role |
|--------|------|
| `src/filtering/learned_filter.py` | Train / infer DeBERTa |
| `src/filtering/nli_filter.py` | Zero-shot NLI |
| `src/filtering/data_split.py` | Base-ID split + test CSV |
| `src/filtering/deberta_filter_evaluator.py` | Threshold + metrics + report DF |
| `src/filtering/config_loader.py` | YAML load / path resolve |
