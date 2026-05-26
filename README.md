# RAG Answer Filtering (Thesis)

Evaluate and improve **answer quality filtering** for RAG systems on ASQA.
The thesis focus is the **filtering layer** — verifying whether a generated
answer is faithful to retrieved context — not building RAG from scratch.

**North-star metric:** minimize **false positive rate (FPR)** subject to
**recall ≥ 0.70**.

## Pipeline

```
retrieve → generate → filter (faithfulness check) → accept / reject
```

NLI framing: **premise = context**, **hypothesis = answer**.

## Repository layout

```
configs/                     # All YAML experiment configs
  filtering/deberta_filter.yaml
  experiments/               # Paths, splits, output dirs
src/rag_filtering/           # Reusable library code
  config/                    # Config loader
  data/                      # ASQA loaders
  rag/                       # RAG baseline support (retrieval + generation)
  filtering/                 # ★ Thesis core — DeBERTa, NLI, ensemble
  evaluation/                # RAGAS + retriever metrics
  utils/
scripts/                     # Headless reproducible runners
notebooks/                   # Thin experiment + reporting layers
data/asqa/                   # Datasets (labeled_asqa.csv is primary)
models/                      # Checkpoints (gitignored)
results/                     # Metrics + prediction CSVs
experiments/                 # Future side experiments
tests/                       # Lightweight sanity tests
```

## Main dataset

`data/asqa/labeled_asqa.csv` — 8,706 balanced pairs (correct + hallucinated)
with gold context. Split by base question ID to prevent leakage.

## Filtering methods

| Method | Module | Role |
|--------|--------|------|
| DeBERTa classifier | `filtering/learned_filter.py` | **Thesis core** |
| NLI zero-shot | `filtering/nli_filter.py` | Required baseline |
| Ensemble | `filtering/ensemble_filter.py` | Meta-classifier |
| LLM judge | `filtering/llm_filter.py` | Supplementary (baseline notebook) |

## Quick start

```bash
pip install -r requirements.txt
pip install -e .

# Train DeBERTa filter (headless)
python scripts/train_filter.py --config configs/experiments/filter_training.yaml

# Select threshold (min FPR @ recall≥0.7) + evaluate on test set
python scripts/evaluate_filter.py --config configs/experiments/filter_training.yaml

# Apply filter to RAG predictions
python scripts/run_filter_on_rag.py --config configs/experiments/asqa_baseline.yaml

# Lexical metric sanity check
python scripts/sanity_check_metrics.py

# Run tests
pytest tests/ -q
```

## Notebooks (workflow order)

| Notebook | Purpose |
|----------|---------|
| `01_asqa_data_preparation.ipynb` | Download/prepare ASQA CSVs |
| `02_filter_training.ipynb` | Train DeBERTa filter + baselines (interactive) |
| `03_rag_asqa_baseline.ipynb` | RAG baseline + filter experiments |
| `04_synthetic_data_generation.ipynb` | Build labeled pairs from RAG output |
| `05_evaluation_analysis.ipynb` | RAGAS comparison + thesis tables |

Notebooks should stay **thin**: import from `rag_filtering`, call functions,
display metrics/plots. Core logic lives in `src/rag_filtering/`.

Every notebook starts with:

```python
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
```

## Scripts vs notebooks

- **Scripts** — reproducible headless runs, write to `results/`
- **Notebooks** — interactive inspection, threshold sweeps, error analysis

## Config files

| File | Purpose |
|------|---------|
| `configs/filtering/deberta_filter.yaml` | Model hyperparams, fp16=false, thresholds |
| `configs/experiments/filter_training.yaml` | Data paths, split ratios, output dirs |
| `configs/experiments/asqa_baseline.yaml` | RAG baseline paths |
| `configs/experiments/ablations.yaml` | Ablation study settings |

Load in code:

```python
from rag_filtering.config.loader import load_yaml, load_config_section
cfg = load_yaml("configs/experiments/filter_training.yaml")
learned = load_config_section("configs/filtering/deberta_filter.yaml", "learned_filter")
```

## Results

| Output | Path |
|--------|------|
| Trained filter | `models/answer_filter/` |
| Threshold selection | `results/threshold_selection.json` |
| Test metrics | `results/learned_filter_test_results.json` |
| RAG predictions | `results/asqa_normal_rag_predictions.csv` |
| Filtered RAG output | `results/rag_predictions_filtered.csv` |

## Experiments

| Experiment | Path | Purpose |
|------------|------|---------|
| Self-RAG verifier | `experiments/self_rag_verifier/` | Flan-T5 generative verifier (side experiment) |
| Self-RAG notebook | `notebooks/06_self_rag_verifier.ipynb` | Interactive train / evaluate / error analysis |

```bash
python experiments/self_rag_verifier/train_verifier.py --train --evaluate --split test
```

See `experiments/self_rag_verifier/README.md`.

**Kaggle:** use `notebooks/06_self_rag_verifier_kaggle.ipynb` (GPU + dataset bootstrap).

## Pre-training gates (DeBERTa filter)

Run in order inside `02_filter_training.ipynb`:

1. Confirm label mapping (`1`=correct, `0`=hallucinated)
2. Inspect paired examples
3. Truncation-collision diagnostic
4. `overfit_sanity_check()` — must reach train F1 ≥ 0.95
5. Full training only after gate passes

See `.cursor/rules/` for full design standards.
