# RAG Evaluation & Answer Filtering

Thesis-focused repository for RAG answer quality evaluation on ASQA.

## Pipeline

```
retrieve → generate → filter (answer quality) → accept/reject
```

The core thesis focus is the **filtering layer**: verifying whether a generated
answer is faithful to retrieved context (NLI-style: premise = context,
hypothesis = answer).

## Project Structure

```
/src
  /data              # ASQA dataset loaders
  /retrieval         # QAPipeline, DocumentIndexer (FAISS)
  /training          # Retriever and generator training
  /filtering         # Answer quality filtering (core thesis focus)
  /evaluation        # RAGAS evaluator, retriever evaluator
  /configs           # YAML configuration files
  /utils             # Model cache, shared helpers
/rag-interference    # Self-RAG-style generative verifier experiment
/data/asqa           # ASQA dataset files
/models              # Saved model checkpoints
/results             # Experiment outputs (regeneratable; not all tracked in git)
/notebooks           # Exploration, training, and analysis
/tests               # Unit tests mirroring /src structure
```

## Quick Start

```bash
pip install -r requirements.txt

# DeBERTa filter training (see notebooks/filter_training.ipynb)
python rag-interference/train_verifier.py --train --evaluate --split test
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `asqa_data_preparation.ipynb` | Prepare ASQA data |
| `rag-asqa-baseline.ipynb` | RAG baseline + filter experiments |
| `filter_training.ipynb` | Train DeBERTa faithfulness classifier |
| `filter_training_kaggle.ipynb` | Kaggle variant of filter training |
| `synthetic_data_generation.ipynb` | Generate labeled pairs from RAG outputs |
| `evaluation_analysis.ipynb` | RAGAS comparison analysis |

## Key Configs

| Config | Path | Purpose |
|--------|------|---------|
| Filtering | `src/configs/filtering.yaml` | DeBERTa faithfulness classifier |
| Verifier | `rag-interference/src/configs/rag_verifier.yaml` | Self-RAG generative verifier |

## Data

Primary dataset: `data/asqa/labeled_asqa.csv` (8,706 samples — balanced correct
and hallucinated pairs with gold context).

See `.cursor/rules/` for design standards and evaluation requirements.
