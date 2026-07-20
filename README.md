# Answer Faithfulness Filtering for RAG

Thesis codebase for **post-generation answer quality filtering** in Retrieval-Augmented Generation (RAG). Retrieval is assumed correct; the filter decides whether a generated answer is **faithful** to the retrieved context (`premise = context`, `hypothesis = answer`).

```text
retrieve → generate → filter(context, answer) → accept / reject
```

**Primary method:** black-box **RAGAS** metrics → sklearn accept/reject classifier (notebooks 3.1 → 3.2).

**Baselines:** LLM-as-judge (notebook 4) and fine-tuned DeBERTa + zero-shot NLI (notebook 5).

**North-star metric:** minimize false-positive rate (accepting hallucinations) subject to recall ≥ 0.70 on correct answers.

Vietnamese install/usage guides (submission style): [`HuongDanCaiDat.txt`](HuongDanCaiDat.txt), [`HuongDanSuDung.txt`](HuongDanSuDung.txt).

## Methods at a glance

| Method | Role | How to run |
|--------|------|------------|
| **RAGAS-feature filter** | **Primary** — extract RAGAS features, train sklearn filter | `notebooks/3.1_ragas-feature-extraction.ipynb` → `3.2_filter-training.ipynb` |
| **LLM-as-judge** | Baseline — OpenAI judge on frozen test | `notebooks/4_llm-judge-filter.ipynb` |
| **DeBERTa / NLI** | Baseline — fine-tuned DeBERTa + zero-shot NLI | `notebooks/5_deberta_nli_baseline.ipynb` or `scripts/run_deberta_nli_baseline.py` |

## Repository layout

Download a pre-trained `models/` directory (optional) from Google Drive and place it at the repo root:

https://drive.google.com/drive/folders/1ZrcY2-_Llr5GJSeZESIXcUmcUWDfjjBc?usp=sharing

```text
ragas-evaluation/
├── configs/
│   ├── filtering/deberta_filter.yaml       # DeBERTa / NLI hyperparams
│   └── experiments/filter_training.yaml    # data paths, seed, n_runs
├── data/
│   ├── labeled_merged.csv                  # ASQA + MS MARCO + WikiEval labels
│   └── labeled_merged_test.csv             # frozen holdout (base-ID split)
├── notebooks/
│   ├── 1_synthetic-data.ipynb
│   ├── 2_rag-asqa-baseline.ipynb
│   ├── 3.1_ragas-feature-extraction.ipynb  # primary: features
│   ├── 3.2_filter-training.ipynb           # primary: train filter
│   ├── 4_llm-judge-filter.ipynb
│   └── 5_deberta_nli_baseline.ipynb
├── scripts/
│   ├── run_deberta_nli_baseline.py
│   ├── train_filter.py / evaluate_filter.py
│   └── package_submission.py
├── src/
│   ├── filtering/                          # core library
│   │   ├── ragas.py                        # RAGAS wrapper (OpenAI)
│   │   ├── ragas_feature_extractor.py
│   │   ├── ragas_filter_trainer.py
│   │   ├── ragas_filter.py
│   │   ├── llm_judge_filter.py
│   │   ├── learned_filter.py              # DeBERTa classifier
│   │   ├── nli_filter.py
│   │   └── deberta_filter_evaluator.py
│   ├── evaluation/
│   └── helper.py
├── models/
│   ├── ragas_filter/                       # sklearn .joblib checkpoints
│   └── answer_filter/                      # DeBERTa runs
└── results/
    ├── ragas_filter/                       # primary experiment outputs
    ├── llm_filter/
    └── deberta_nli/
```

## Setup

```bash
git clone https://github.com/Htaaxx/ragas-evaluation.git
cd ragas-evaluation

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt

# Optional — GPU torch for DeBERTa training:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Create a `.env` in the repo root (there is no `.env_template`):

```bash
OPENAI_API_KEY=sk-...
# optional:
# HUGGINGFACE_API_KEY=...
# GOOGLE_API_KEY=...
```

`OPENAI_API_KEY` is required for RAGAS feature extraction (3.1) and the LLM-judge baseline (4). DeBERTa / NLI do not need OpenAI.

Pinned stack highlights: `ragas==0.1.9`, LangChain 0.2.x, `transformers==4.40.2`.

## Data and splits

| File | Role |
|------|------|
| `data/labeled_merged.csv` | Full labeled corpus (~9.8k rows): ASQA, MS MARCO, WikiEval |
| `data/labeled_merged_test.csv` | Frozen holdout for LLM-judge and DeBERTa / NLI |

Schema: `id, question, context, answer, label, dataset`. Hallucinated pairs use the `_hallu` suffix (e.g. `asqa_0` / `asqa_0_hallu`).

DeBERTa splits are **by base question ID** so correct/hallucinated pairs never cross train/val/test (`test_size=0.2`, `seed=42`, `reuse_frozen_test: true` in [`configs/experiments/filter_training.yaml`](configs/experiments/filter_training.yaml)).

RAGAS notebook 3.2 uses `train_test_split(test_size=0.2, random_state=42)` on feature rows — a separate protocol from the frozen DeBERTa test CSV.

---

## Primary method: RAGAS-feature filter (notebooks 3.1–3.2)

Black-box RAGAS scores (no gold answer at inference) become features for a supervised sklearn filter.

### 3.1 Feature extraction

Open [`notebooks/3.1_ragas-feature-extraction.ipynb`](notebooks/3.1_ragas-feature-extraction.ipynb), restart the kernel, run all cells.

| Item | Value |
|------|--------|
| Input | `data/labeled_merged.csv` |
| API | `OPENAI_API_KEY` |
| LLM / embeddings | `gpt-4o-mini` + `text-embedding-3-small`, `temperature=0` |
| Metrics used | `faithfulness`, `answer_relevancy`, `context_relevancy` |
| Metrics skipped | `answer_correctness`, `answer_similarity` (need gold) |
| Runs | `N_TIMES=3` (repeat for stability) |

**Outputs:**

```text
results/ragas_filter/merged/merged_ragas_features_{1,2,3}.csv
results/ragas_filter/merged/merged_ragas_checkpoints_{1,2,3}.csv
```

This step calls the OpenAI API — expect time and cost on the full ~9.8k rows. Prefer a small subset first.

Library entry points: `src/filtering/ragas.py`, `ragas_feature_extractor.py`.

### 3.2 Filter training

Open [`notebooks/3.2_filter-training.ipynb`](notebooks/3.2_filter-training.ipynb), restart the kernel, run all cells.

- Loads features from 3.1 + labels from `labeled_merged.csv`
- Trains sklearn pipelines (logistic regression, random forest, gradient boosting, ExtraTrees, HistGB, optional XGBoost)
- Selects and saves the best models

**Outputs:**

```text
models/ragas_filter/*.joblib
results/ragas_filter/merged/average_results.csv
results/ragas_filter/merged/summary_classification_report.csv
results/ragas_filter/ragas_filter_merged_summary.csv
```

Library entry points: `src/filtering/ragas_filter_trainer.py`, `ragas_filter.py`.

---

## Baseline: LLM-as-judge (notebook 4)

Open [`notebooks/4_llm-judge-filter.ipynb`](notebooks/4_llm-judge-filter.ipynb).

| Item | Value |
|------|--------|
| Input | `data/labeled_merged_test.csv` |
| API | `OPENAI_API_KEY` |
| Model | `gpt-4o-mini` (notebook default) |
| Class | `LLMJudgeFilter` in `src/filtering/llm_judge_filter.py` |

Typically runs three prediction passes, then writes a summary.

**Outputs:**

```text
results/llm_filter/classification/predictions_run_{1,2,3}.csv
results/llm_filter/classification/llm_judge_summary.csv
```

---

## Baseline: DeBERTa / NLI (notebook 5)

Fine-tune `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` as a binary faithfulness classifier, pick a validation threshold with `select_threshold_min_fpr(..., min_recall=0.70)`, evaluate on the frozen test CSV, and compare to zero-shot NLI and a no-filter baseline.

### Headless

```bash
python scripts/run_deberta_nli_baseline.py --config configs/experiments/filter_training.yaml
```

Resume NLI only (skip DeBERTa training):

```bash
python scripts/run_deberta_nli_baseline.py \
  --config configs/experiments/filter_training.yaml \
  --skip-train --skip-overfit-gate
```

### Interactive

Open [`notebooks/5_deberta_nli_baseline.ipynb`](notebooks/5_deberta_nli_baseline.ipynb): Setup → optional gates → Train.

### Config

- [`configs/experiments/filter_training.yaml`](configs/experiments/filter_training.yaml) — paths, `n_runs: 1`, `reuse_frozen_test: true`
- [`configs/filtering/deberta_filter.yaml`](configs/filtering/deberta_filter.yaml) — model hyperparams

**Important:** keep `fp16: false` (DeBERTa-v3 instability). Prefer GPU ≥ 4 GB VRAM; default batch settings target laptop GPUs.

### Artifacts

```text
models/answer_filter/run_1/
results/deberta_nli/run_1/
results/deberta_nli/nli_zeroshot/
results/deberta_nli/summary.json
results/deberta_nli/summary_classification_report.csv
```

A row with precision ≈ 0.5 and recall = 1.0 is usually the **no-filter** baseline, not a collapsed model.

---

## Notebook order

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `1_synthetic-data.ipynb` | Hallucinated answers / labeling support |
| 2 | `2_rag-asqa-baseline.ipynb` | Normal RAG predictions |
| 3.1 | `3.1_ragas-feature-extraction.ipynb` | **Primary** — RAGAS feature tables |
| 3.2 | `3.2_filter-training.ipynb` | **Primary** — train RAGAS-feature classifiers |
| 4 | `4_llm-judge-filter.ipynb` | LLM-as-judge baseline |
| 5 | `5_deberta_nli_baseline.ipynb` | DeBERTa + NLI baseline |

## Evaluation conventions

Report for every filter: accuracy, precision, recall, F1, ROC-AUC, acceptance rate, and confusion / FPR. Shared report columns:

```text
dataset,num_samples,accepted,acceptance_rate,accuracy,precision,recall,f1,roc_auc
```

Do **not** use argmax at 0.5 as the final DeBERTa decision rule — always use the validation-selected threshold.

## Library imports

From the repo root (notebooks typically `sys.path.append('..')`):

```python
from src.filtering.ragas_feature_extractor import RagasFeatureExtractor
from src.filtering.ragas_filter_trainer import RagasFilterTrainer
from src.filtering.ragas_filter import RagasFilter
from src.filtering.llm_judge_filter import LLMJudgeFilter
from src.filtering.learned_filter import AnswerQualityClassifier
from src.filtering.nli_filter import NLIAnswerFilter
```

## Tests

```bash
python -m pytest tests/test_data_split.py tests/test_imports.py tests/test_filter_metrics.py -v
```

## Documentation

| Doc | Contents |
|-----|----------|
| [HuongDanCaiDat.txt](HuongDanCaiDat.txt) | Vietnamese install guide |
| [HuongDanSuDung.txt](HuongDanSuDung.txt) | Vietnamese usage guide |

This README is the full English guide (setup, RAGAS primary path, LLM-judge, DeBERTa/NLI).

## License / academic use

Research / thesis project. Cite upstream models and datasets you use (ASQA, MS MARCO, WikiEval, DeBERTa-v3 MNLI, RAGAS, OpenAI models).
