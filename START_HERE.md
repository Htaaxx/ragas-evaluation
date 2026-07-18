# Start here — thesis filtering pipeline

This repo studies **whether a RAG answer is faithful to its retrieved context**. Work through notebooks in order, then run the DeBERTa baseline for thesis comparison tables.

## 1. Environment

```bash
cd ragas-evaluation
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
# For DeBERTa training on NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Add API keys to `.env` if you will run RAGAS LLM metrics or the LLM-judge notebook.

## 2. Data you already have

| File | Use |
|------|-----|
| `data/labeled_merged.csv` | Full labeled corpus (train/val/test source) |
| `data/labeled_merged_test.csv` | Frozen holdout from base-ID split (`test_size=0.2`, `seed=42`) |
| `data/asqa/`, `data/ms-marco/`, `data/wikiEval/` | Per-dataset sources |

Hallucinated rows use ids like `asqa_0_hallu`. Splitting always groups by base id so correct and hallu twins stay in the same fold.

## 3. Recommended path

### A. Explore data and RAGAS filter (primary method)

1. `notebooks/0_data_collection.ipynb`
2. `notebooks/1_synthetic-data.ipynb`
3. `notebooks/2_rag-asqa-baseline.ipynb`
4. `notebooks/3.1_ragas-feature-extraction.ipynb`
5. `notebooks/3.2_filter-training.ipynb`
6. `notebooks/4_llm-judge-filter.ipynb` (optional baseline)

Existing RAGAS outputs live under `results/ragas_filter/`.

### B. DeBERTa / NLI baseline (thesis baseline)

Interactive:

1. Open `notebooks/5_deberta_nli_baseline.ipynb`
2. Run setup + split + gates
3. Set `RUN_TRAINING = True` **or** run the headless script below

Headless (preferred for the 3-run protocol):

```bash
python scripts/run_deberta_nli_baseline.py --config configs/experiments/filter_training.yaml
```

This will:

1. Rebuild the leakage-safe split and refresh `data/labeled_merged_test.csv`
2. Run the overfit sanity gate (must reach train F1 ≥ 0.95)
3. Train DeBERTa three times → `models/answer_filter/run_{1,2,3}/`
4. Pick min-FPR thresholds on val (recall ≥ 0.70)
5. Evaluate on the frozen test set
6. Run zero-shot NLI once
7. Write `results/deberta_nli/summary.json` with mean±std

Resume evaluation only (checkpoints already trained):

```bash
python scripts/run_deberta_nli_baseline.py --skip-train --skip-overfit-gate
```

## 4. What “good” looks like

- Filter framing is always `(context, answer)`, not `(question, answer)`.
- Final decisions use the **validation-selected threshold**, never argmax@0.5.
- Comparison tables include **No Filter**, **NLI zero-shot**, and **DeBERTa (mean±std)**.
- Keep `fp16: false` for DeBERTa-v3.

## 5. Quick smoke tests

```bash
python -m pytest tests/test_data_split.py tests/test_imports.py tests/test_filter_metrics.py -v
```

## 6. Where logic lives

Notebooks are thin. Core code is under `src/filtering/`:

- `learned_filter.py` — DeBERTa train / infer
- `nli_filter.py` — zero-shot NLI
- `data_split.py` — base-ID split + test CSV export
- `deberta_filter_evaluator.py` — metrics + `select_threshold_min_fpr`
- `ragas_*.py` — RAGAS-feature filter stack

See the root `README.md` for the full thesis overview.

## 7. Further reading

| Doc | When to open it |
|-----|-----------------|
| [docs/DATA_AND_SPLITS.md](docs/DATA_AND_SPLITS.md) | Schema, `_hallu` pairs, how the frozen test CSV is built |
| [docs/EVALUATION.md](docs/EVALUATION.md) | North-star threshold rule and shared metrics table |
| [docs/DEBERTA_NLI_BASELINE.md](docs/DEBERTA_NLI_BASELINE.md) | Kaggle steps, gates, and which files to download after training |
