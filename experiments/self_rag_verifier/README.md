# Self-RAG Generative Answer Verifier

Side experiment: fine-tune **Flan-T5** to generate Self-RAG reflection tokens
instead of using the thesis-core DeBERTa classification head.

**Not the thesis-core path** — the main filter lives in
`src/rag_filtering/filtering/learned_filter.py`.

## What it does

- **Input:** `(question, gold_context, candidate_answer)`
- **Output:** `[IsRel] relevant [IsSup] fully_supported [IsUse] 5 [Decision] ACCEPT`
- **No retriever** — uses gold context from `labeled_asqa.csv`

## Files

```
notebooks/06_self_rag_verifier.ipynb         # Interactive train / evaluate / analysis
notebooks/06_self_rag_verifier_kaggle.ipynb  # Kaggle-ready (GPU bootstrap + output zip)
experiments/self_rag_verifier/
  verifier_system.py    # VerifierSystem + _VerifierDataset
  train_verifier.py     # CLI entry point
configs/experiments/
  rag_verifier.yaml     # Model, training, prompt, reflection tokens
```

Reuses from the main package:

- `rag_filtering.filtering.data_split.load_and_split`
- `rag_filtering.filtering.learned_filter._extract_top1_context`
- `rag_filtering.config.loader.load_yaml` / `resolve_path`

## Quick start

From the repository root:

```bash
pip install -r requirements.txt
pip install -e .

# Train + evaluate on test split
python experiments/self_rag_verifier/train_verifier.py --train --evaluate --split test

# Evaluate existing checkpoint only (requires a trained checkpoint)
python experiments/self_rag_verifier/train_verifier.py --evaluate --split test
```

**HuggingFace auth:** The base model (`google/flan-t5-base`) is downloaded on first
train. If you see `401 Unauthorized` / `User Access Token ... is expired`, either
refresh your token (`huggingface-cli login`) or rely on the fix that passes
`token=False` for public models (already in `verifier_system.py`).

**Checkpoint path:** Save dir is `models/answer_verifier/` (not `rag_verifier`) —
paths containing the substring `rag` break Transformers `AutoConfig` heuristics on
empty directories.

## Outputs

| Output | Path |
|--------|------|
| Checkpoint | `models/answer_verifier/` |
| Metrics JSON | `results/answer_verifier/metrics_test.json` |

## Config

Edit `configs/experiments/rag_verifier.yaml` for model name, epochs, learning rate,
prompt template, and reflection token targets.

## Kaggle

Use `notebooks/06_self_rag_verifier_kaggle.ipynb`.

1. Create a Kaggle dataset with `labeled_asqa.csv` (slug e.g. `ragas-data`).
2. New notebook → upload or copy the Kaggle notebook.
3. Settings: **GPU T4 x2**, **Internet On**.
4. Add Input: your data dataset (code is cloned from GitHub by default).
5. Edit cell 0 if paths differ:
   - `LABELED_CSV_INPUT = "/kaggle/input/<your-data-slug>/labeled_asqa.csv"`
6. Run All → **Save Version** → download zips from Output tab.
