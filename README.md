# Answer Faithfulness Filtering for RAG

Thesis codebase for **post-generation answer quality filtering** in Retrieval-Augmented Generation (RAG). Retrieval is assumed correct; the filter decides whether a generated answer is **faithful** to the retrieved context (NLI framing: premise = context, hypothesis = answer).

**North-star metric:** minimize false-positive rate (accepting hallucinations) subject to recall ≥ 0.70 on correct answers.

## Pipeline

```text
retrieve → generate → filter (context, answer) → accept / reject
```

Two complementary filter families are implemented:

| Method | Role | Notebook / script |
|--------|------|-------------------|
| **DeBERTa / NLI** | Thesis **baseline** — fine-tuned DeBERTa binary classifier + zero-shot NLI | `notebooks/5_deberta_nli_baseline.ipynb`, `scripts/run_deberta_nli_baseline.py` |
| **RAGAS-feature filter** | Primary method — black-box RAGAS metrics → sklearn classifiers | `notebooks/3.1_*`, `3.2_*` |
| **LLM-as-judge** | Additional baseline | `notebooks/4_llm-judge-filter.ipynb` |

## Repository layout

```text
ragas-evaluation/
├── configs/
│   ├── filtering/deberta_filter.yaml      # DeBERTa / NLI hyperparams
│   └── experiments/filter_training.yaml # split paths, seeds, n_runs
├── data/
│   ├── labeled_merged.csv               # ASQA + MS MARCO + WikiEval labels
│   └── labeled_merged_test.csv          # frozen holdout (base-ID split, seed=42)
├── notebooks/
│   ├── 0_data_collection.ipynb
│   ├── 1_synthetic-data.ipynb
│   ├── 2_rag-asqa-baseline.ipynb
│   ├── 3.1_ragas-feature-extraction.ipynb
│   ├── 3.2_filter-training.ipynb
│   ├── 4_llm-judge-filter.ipynb
│   └── 5_deberta_nli_baseline.ipynb     # DeBERTa/NLI baseline (×3 runs)
├── scripts/
│   ├── run_deberta_nli_baseline.py      # headless DeBERTa ×3 + NLI
│   ├── train_filter.py / evaluate_filter.py
│   └── run_filter_on_rag.py
├── src/
│   ├── filtering/                       # core library
│   │   ├── learned_filter.py            # AnswerQualityClassifier + train_classifier
│   │   ├── nli_filter.py                # NLIAnswerFilter
│   │   ├── data_split.py                # leakage-safe base-ID split
│   │   ├── deberta_filter_evaluator.py  # min-FPR threshold + metrics
│   │   ├── ragas_*.py                   # RAGAS-feature pipeline
│   │   └── llm_judge_filter.py
│   ├── evaluation/                      # shared evaluators / plots
│   └── utils/
├── models/answer_filter/run_{1,2,3}/    # DeBERTa checkpoints
└── results/deberta_nli/                 # thresholds, per-run metrics, summary.json
```

## Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
# GPU training (recommended for DeBERTa):
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Copy `.env` with any LLM API keys needed for RAGAS / LLM-judge notebooks.

## Data and splits

Primary labeled corpus: `data/labeled_merged.csv` (~9.8k rows, balanced correct / hallucinated; datasets: ASQA, MS MARCO, WikiEval).

Hallucinated IDs use the `_hallu` suffix (e.g. `asqa_0` / `asqa_0_hallu`). Splits are **by base question ID** so pairs never cross train/val/test:

- `test_size = 0.2`, `random_state = 42`
- validation carved from the remaining train IDs (`val_ratio = 0.2`)
- held-out test persisted to `data/labeled_merged_test.csv`

Configured in `configs/experiments/filter_training.yaml`.

## DeBERTa / NLI baseline (notebook 5)

1. Pre-training gates (label check, pair spot-check, truncation diagnostic, overfit sanity check).
2. Fine-tune `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` three times on the same frozen split.
3. Select threshold on validation with `select_threshold_min_fpr(..., min_recall=0.70)`.
4. Evaluate on the frozen test CSV; aggregate mean±std.
5. Compare against zero-shot NLI and a no-filter baseline.

Headless:

```bash
python scripts/run_deberta_nli_baseline.py --config configs/experiments/filter_training.yaml
```

Artifacts:

- `models/answer_filter/run_{1,2,3}/`
- `results/deberta_nli/run_{1,2,3}/` (threshold, metrics, predictions)
- `results/deberta_nli/nli_zeroshot/`
- `results/deberta_nli/summary.json`

**Important:** keep `fp16: false` in `deberta_filter.yaml` (DeBERTa-v3 instability). Prefer GPU with ≥4 GB VRAM; `batch_size: 1` + gradient accumulation targets 4 GB laptop GPUs.

## RAGAS-feature filter (notebooks 3.1–3.2)

Extract black-box RAGAS features (no gold answer at inference), train sklearn classifiers, and evaluate classification quality. See `START_HERE.md` for the RAGAS-oriented walkthrough and `results/ragas_filter/` for existing experiment outputs.

## Notebook order

| # | Notebook | Purpose |
|---|----------|---------|
| 0 | Data collection | ASQA / MS MARCO / WikiEval sources |
| 1 | Synthetic data | Hallucinated answers / labeling |
| 2 | RAG ASQA baseline | Normal RAG predictions |
| 3.1 | RAGAS feature extraction | Feature tables for merged data |
| 3.2 | Filter training | Train RAGAS-feature classifiers |
| 4 | LLM-judge filter | LLM-as-judge baseline |
| 5 | DeBERTa / NLI baseline | Fine-tuned + zero-shot NLI (×3) |

## Evaluation conventions

Every filter comparison should report precision, recall, F1, accuracy, FPR, rejection recall / rate, and a confusion matrix. Required baselines in DeBERTa experiments: **No Filter**, **NLI zero-shot**, **fine-tuned DeBERTa**.

Do not use argmax at 0.5 as the final decision rule — always use the validation-selected threshold.

## Tests

```bash
python -m pytest tests/test_data_split.py tests/test_imports.py tests/test_filter_metrics.py -v
```

## License / academic use

Research / thesis project. Cite the upstream models and datasets you use (ASQA, MS MARCO, WikiEval, DeBERTa-v3 MNLI, RAGAS).
