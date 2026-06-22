# Self-RAG Inference Generator

Standalone answer-generation experiment for MS MARCO. This folder is separate
from `experiments/self_rag_verifier/`, which remains the ACCEPT/REJECT verifier.

## What This Pipeline Does

```text
MS MARCO CSV -> split context passages -> FAISS retrieval -> Self-RAG 7B
             -> reflection-token scoring -> best answer -> EM/F1/ROUGE-L
```

- Input: `data/ms-macro/msmarcro_500.csv`
- Retriever: `sentence-transformers/all-MiniLM-L6-v2` + FAISS
- Generator: `selfrag/selfrag_llama2_7b`
- Primary runtime: Kaggle GPU T4 x2 with the HuggingFace 4-bit fallback
- Outputs: `results/self_rag_inference/msmarco_predictions.csv` and
  `results/self_rag_inference/metrics.json`

## Local Retrieval Smoke Test

The 7B Self-RAG model is not expected to run on a CPU Windows machine. Locally,
validate the data and retriever wiring:

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
- Install: `pip install accelerate bitsandbytes sentence-transformers faiss-cpu rouge-score`
- Backend: `model.backend: hf`
- Quantization: `hf_fallback.load_in_4bit: true`

Do not install vLLM in Kaggle's shared environment by default. Recent vLLM
wheels can upgrade CUDA/numpy/protobuf/starlette packages and trigger resolver
conflicts with Kaggle's preinstalled RAPIDS, TensorFlow, Colab, and Google
packages. Use vLLM only in a clean Linux environment or if you are prepared to
manage those dependency changes manually.

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

- `model.name`: `selfrag/selfrag_llama2_7b`
- `model.backend`: `hf` or `vllm`
- `retriever.top_k`: number of retrieved passages
- `data.max_samples`: optional row limit for debugging
- `generation.score_weights`: weights for `[Relevant]`, `[Fully supported]`,
  and `[Utility:n]` reflection tokens

## Notes

- This experiment does not train Self-RAG. It performs inference with the
  official pretrained Self-RAG checkpoint.
- The prompt uses the official shape:

```text
### Instruction:
{question}

### Response:
[Retrieval]<paragraph>{paragraph}</paragraph>
```

- `skip_special_tokens=False` is required so reflection tokens remain visible
  for parsing and scoring.
