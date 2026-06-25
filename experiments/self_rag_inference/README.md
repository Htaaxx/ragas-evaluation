# Self-RAG Inference Generator

Standalone answer-generation experiment for MS MARCO. This folder is separate
from `experiments/self_rag_verifier/`, which remains the ACCEPT/REJECT verifier.

## What This Pipeline Does

```text
MS MARCO CSV -> split context passages -> FAISS retrieval -> Qwen instruct generator
             -> best answer -> EM/F1/ROUGE-L
```

- Input: `data/ms-macro/msmarcro_500.csv`
- Retriever: `sentence-transformers/all-MiniLM-L6-v2` + FAISS
- Generator: `Qwen/Qwen2.5-7B-Instruct`
- Primary runtime: Kaggle GPU T4 x2 with HuggingFace 4-bit causal generation
- Outputs: `results/self_rag_inference/msmarco_predictions.csv` and
  `results/self_rag_inference/metrics.json`

## Local Retrieval Smoke Test

The 7B generator is not expected to run comfortably on a CPU Windows machine.
Locally, validate the data and retriever wiring:

```bash
python experiments/self_rag_inference/run_inference.py --retrieval-only --limit 5
```

This builds or loads:

```text
results/self_rag_inference/msmarco_faiss_index/
```

and writes:

```text
results/self_rag_inference/retrieval_preview.csv
```

## Kaggle T4 x2 Full Run

Use `notebooks/07_self_rag_inference_kaggle.ipynb`.

Recommended Kaggle settings:

- Accelerator: GPU T4 x2
- Internet: On
- Install: `pip install accelerate bitsandbytes sentencepiece sentence-transformers faiss-cpu rouge-score`
- Backend: `model.backend: causal_instruct`
- Generator: `model.name: Qwen/Qwen2.5-7B-Instruct`
- Quantization: `model.load_in_4bit: true`

Do not install vLLM in Kaggle's shared environment by default. Recent vLLM
wheels can upgrade CUDA/numpy/protobuf/starlette packages and trigger resolver
conflicts with Kaggle's preinstalled RAPIDS, TensorFlow, Colab, and Google
packages. The old `selfrag/selfrag_llama2_7b` backend is kept for experiments,
but it produced poor MS MARCO QA outputs through the Kaggle HF 4-bit path.

The notebook runs the same CLI:

```bash
python experiments/self_rag_inference/run_inference.py \
  --config configs/experiments/self_rag_inference.yaml
```

## Config

Main config:

```text
configs/experiments/self_rag_inference.yaml
```

Important knobs:

- `model.name`: `Qwen/Qwen2.5-7B-Instruct`
- `model.backend`: `causal_instruct`, `seq2seq`, `hf`, or `vllm`
- `retriever.top_k`: number of retrieved passages
- `data.max_samples`: optional row limit for debugging
- `model.max_input_tokens`: context prompt length budget
- `model.max_new_tokens`: answer generation budget

## Notes

- This experiment does not train a generator. It performs inference with a
  pretrained instruction model.
- The default prompt is normal RAG QA:

```text
Answer the question using only the context below. If the context does not contain
the answer, say "I don't know." Give a short, direct answer.

Question: {question}

Context:
{context}

Answer:
```
- Self-RAG reflection parsing is still available for legacy `hf` / `vllm`
  backends, but `causal_instruct` is the recommended backend for MS MARCO QA.
