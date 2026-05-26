# Self-RAG Answer Verifier

Generative answer verification experiment using Flan-T5 and Self-RAG reflection tokens.

## Scope

- **Input**: `(question, gold_context, candidate_answer)`
- **Output**: `[IsRel] ... [IsSup] ... [IsUse] ... [Decision] ACCEPT|REJECT`
- **No retriever** — gold context from `labeled_asqa.csv` is used directly.

## Layout

```
rag-interference/
  train_verifier.py          # CLI entry point
  src/
    verifier_system.py       # VerifierSystem + _VerifierDataset
    configs/
      rag_verifier.yaml      # Model, data, training, prompt config
```

Reuses shared utilities from the main repo:

- `src/filtering/data_split.py` — leakage-safe train/val/test split
- `src/filtering/learned_filter.py` — `_extract_top1_context()`

## Quick start

From the repository root:

```bash
# Train + evaluate on test split
python rag-interference/train_verifier.py --train --evaluate --split test

# Evaluate an existing checkpoint only
python rag-interference/train_verifier.py --evaluate --split test
```

Outputs:

- Checkpoint: `models/rag_verifier/`
- Metrics JSON: `results/rag_verifier/metrics_test.json`

## Config

All tunables live in `rag-interference/src/configs/rag_verifier.yaml`:

- Model: `google/flan-t5-base`
- Reflection targets for label=1 (ACCEPT) and label=0 (REJECT)
- Prompt template with `{question}`, `{context}`, `{answer}`
