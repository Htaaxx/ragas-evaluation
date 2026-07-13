# Normal RAG Inference

Dataset-agnostic retrieve-then-generate QA pipeline. The main/default dataset is
the merged thesis dataset, while MS MARCO is kept as a secondary benchmark.

## Pipeline

```text
CSV -> parse context passages -> FAISS retrieval -> Qwen 4-bit generation
    -> answer cleanup -> metrics + diagnostics
```

## Main Run

```bash
python experiments/normal_rag_inference/run_inference.py \
  --config configs/experiments/normal_rag_merged.yaml
```

Input: `data/merged/labeled_merged.csv`

Outputs:

```text
results/normal_rag/merged/predictions.csv
results/normal_rag/merged/metrics.json
```

## Kaggle T4 x2

Use `notebooks/08_normal_rag_inference_kaggle.ipynb` (**merged-only** on
`final-filtering-pipeline`; MS MARCO config was removed).

Recommended Kaggle settings:

- Accelerator: GPU T4 x2
- Internet: On
- Install: `pip install accelerate bitsandbytes sentencepiece sentence-transformers faiss-cpu rouge-score`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Backend: `causal_instruct`
- Quantization: 4-bit

## Prompt

The prompt is normal RAG and intentionally does not tell the model to abstain:

```text
Answer the question using the retrieved context below. Give a short, direct answer.

Question: {question}

Context:
{context}

Answer:
```

For this benchmark, every row should get an answer. Use diagnostics in
`metrics.json` to catch empty answers, abstentions, or leaked chat tokens.
