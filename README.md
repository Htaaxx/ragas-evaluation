# 🔍 RAG System - Retrieval-Augmented Generation

Hệ thống QA hoàn chỉnh kết hợp **tìm kiếm tài liệu** (Retrieval) + **sinh câu trả lời** (Generation) dựa trên HotpotQA.

**Features:**
- ✅ **Retriever**: Fine-tune sentence-transformers trên cặp câu hỏi-đoạn văn
- ✅ **Generator**: Fine-tune mô hình seq2seq (T5) với contexts được lấy
- ✅ **Indexing**: FAISS vector index cho tìm kiếm nhanh
- ✅ **QA Pipeline**: Lấy tài liệu → Sinh câu trả lời (end-to-end)
- ✅ **Model Caching**: Tự động tải models từ HuggingFace và cache locally
- ✅ **GPU Support**: Mixed precision (FP16), gradient accumulation

---

## 📦 Cài Đặt

### 1. Clone Repository & Cài Dependencies

```bash
# Clone repo
git clone <your-repo-url>
cd "RAG with HotpotQA"

# Tạo virtual environment (optional nhưng recommended)
python -m venv venv

# Kích hoạt
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt

# Nếu dùng GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Tạo File `.env` (Optional - HuggingFace Token)

```bash
# Tạo file .env trong root directory
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

**Cách lấy token:**
1. Truy cập [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Tạo token mới (Read access là đủ)
3. Copy vào `.env`

> **Ghi chú**: Không bắt buộc nếu dùng public models

### 3. Chuẩn Bị Dữ Liệu (HotpotQA)

```bash
# Download HotpotQA từ official source
# hoặc đặt file CSV vào: data/hotpot_qa/
#   - train.csv
#   - valid.csv
```

**CSV Format:**
```csv
question,answer,context,supporting_facts
"Where is the Eiffel Tower?","Paris, France","{...context JSON...}","{...supporting facts...}"
```

---

## 🚀 Quick Start

### Ví dụ 1: QA đơn giản (không training)

```python
from notebooks.rag import RAGSystem

# Khởi tạo (models auto-download & cache)
rag = RAGSystem()

# Tạo index từ documents
docs = [
    "The Eiffel Tower is in Paris, France.",
    "The Colosseum is in Rome, Italy.",
    "The Statue of Liberty is in New York, USA.",
]
rag.index_corpus(docs)

# Lưu index (dùng lại sau)
rag.save_index("./rag_output/index")

# Trả lời câu hỏi
answer = rag.answer("Where is the Eiffel Tower?")
print(answer)  # "Paris, France" hoặc tương tự
```

### Ví dụ 2: Training Retriever (Full Pipeline)

```python
from notebooks.rag import RAGSystem
import os

# Setup HuggingFace token
os.environ["HF_TOKEN"] = "hf_..."

# Khởi tạo
rag = RAGSystem(
    encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    generator_model="google/flan-t5-base",
    output_dir="./rag_output",
    models_dir="../models",
    device="cuda"  # hoặc "cpu"
)

# 1. Tải dữ liệu HotpotQA
rag.load_hotpotqa_data(
    train_path="data/hotpot_qa/train.csv",
    valid_path="data/hotpot_qa/valid.csv"
)

# 2. Build corpus
rag.build_corpus()

# 3. Training retriever
rag.create_retriever_train_loader(batch_size=16)

print("\n🚀 Training retriever...")
rag.train_retriever(
    epochs=5,
    lr=2e-5,
    patience=3,
    save_name="retriever_trained"
)

# 4. Evaluate retriever
print("\n📊 Evaluating retriever...")
metrics = rag.evaluate_retriever(top_k=5)
# Prints: top1_acc, top3_acc, top5_acc, mrr

# 5. Indexing cho QA
print("\n📦 Building FAISS index...")
rag.index_corpus(rag.corpus_texts, batch_size=64)
rag.save_index("./rag_output/index")

# 6. Testing
question = "What is the capital of France?"
answer = rag.answer(question, top_k=3)
print(f"Q: {question}")
print(f"A: {answer}")
```

### Ví dụ 3: Training Generator

```python
from notebooks.rag import RAGSystem, TrainExample

rag = RAGSystem()

# Tải dữ liệu + indexing
rag.load_hotpotqa_data("data/hotpot_qa/train.csv", "data/hotpot_qa/valid.csv")
rag.build_corpus()
rag.index_corpus(rag.corpus_texts)

# Tạo training examples từ HotpotQA
train_examples = []
for _, row in rag.df_train.iterrows():
    train_examples.append(
        TrainExample(
            question=row["question"],
            answer=row["answer"],
            contexts=None  # Auto-retrieve
        )
    )

# Training generator
print("\n🚀 Training generator...")
rag.train_generator(
    train_examples[:1000],  # First 1000 examples
    batch_size=4,
    epochs=3,
    lr=5e-5,
    max_input_tokens=512,
    max_target_tokens=128,
    top_k=5,
    save_name="generator_trained"
)

# Test
answer = rag.answer("What is AI?", top_k=5, max_new_tokens=64)
print(f"Answer: {answer}")
```

---

## 📂 Cấu Trúc Thư Mục

```
RAG with HotpotQA/
│
├── notebooks/
│   ├── rag.py                      # ⭐ Core RAGSystem class
│   ├── rag-hotpotqa.ipynb          # Jupyter notebook demo
│   ├── rag_example.py              # Usage examples
│   ├── data_collection.ipynb       # Data preprocessing
│   └── verbalization.ipynb         # Text utilities
│
├── data/
│   └── hotpot_qa/
│       ├── train.csv               # ~90k samples
│       ├── valid.csv               # ~5.9k samples
│       ├── valid_async.csv         # Alternative validation
│       └── valid_async_checkpoint.csv
│
├── models/                         # 🔄 Auto-created, models cached here
│   ├── sentence-transformers--all-MiniLM-L6-v2/
│   ├── google--flan-t5-base/
│   └── ...
│
├── rag_output/                     # 📊 Outputs auto-created
│   ├── index/
│   │   ├── index.faiss
│   │   └── meta.json
│   ├── embeddings.pt
│   └── ...
│
├── requirements.txt                # Python packages
├── README.md                       # This file
├── ARCHITECTURE.md                 # 📋 System architecture
└── .env                           # Environment variables (token)
```

**Ghi chú:** `models/` và `rag_output/` tự động tạo khi chạy code

---

## 🔧 Configuration

### Environment Variables

```bash
# .env file
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
HF_HOME=/path/to/custom/cache  # (optional) Custom HF cache
```

### RAGSystem Parameters

```python
RAGSystem(
    # Models
    encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    generator_model="google/flan-t5-base",
    
    # Paths
    index_dir=None,              # Pre-built index path
    output_dir="./rag_output",   # Output directory
    models_dir="../models",      # Model cache directory
    
    # Hardware
    device="cuda",               # "cuda" hoặc "cpu"
    
    # Auth
    hf_token=None,               # HF token (auto from .env)
)
```

---

## 📚 API Reference

### Core Classes & Methods

#### **RAGSystem Class**

```python
# Initialization
rag = RAGSystem(encoder_model="...", generator_model="...", device="cuda")

# Data Loading
rag.load_hotpotqa_data(train_path, valid_path)
rag.build_corpus(df=None)

# Retriever Training
rag.create_retriever_train_loader(batch_size=16, shuffle=True)
rag.train_retriever(epochs=5, lr=2e-5, patience=3, save_name="...")
rag.evaluate_retriever(top_k=5, rebuild_cache=False)

# Indexing
rag.index_corpus(documents, batch_size=64)
rag.save_index(index_dir)
rag.load_index(index_dir)

# Generator Training
rag.train_generator(train_examples, batch_size=4, epochs=3, ...)

# QA
rag.retrieve(query, top_k=5) → list[str]
rag.answer(question, top_k=5) → str
```

#### **Data Classes**

```python
# Training example (cho generator)
TrainExample(
    question: str,
    answer: str,
    contexts: Sequence[str] | None = None
)

# Retriever training pair
RetrieverExample(
    question: str,
    positive_passage: str,
    negative_passages: Sequence[str] | None = None
)
```

---

## 🎯 Complete Workflow Example

```python
#!/usr/bin/env python3
"""Full RAG pipeline: training + evaluation + QA"""

import os
from notebooks.rag import RAGSystem, TrainExample

def main():
    # Setup
    os.environ["HF_TOKEN"] = "hf_..."
    
    # 1️⃣ Initialize
    rag = RAGSystem(
        device="cuda",
        output_dir="./rag_output",
        models_dir="../models"
    )
    
    # 2️⃣ Load data
    rag.load_hotpotqa_data(
        "data/hotpot_qa/train.csv",
        "data/hotpot_qa/valid.csv"
    )
    rag.build_corpus()
    
    # 3️⃣ Train retriever
    rag.create_retriever_train_loader(batch_size=16)
    rag.train_retriever(epochs=3)
    rag.evaluate_retriever()
    
    # 4️⃣ Index corpus
    rag.index_corpus(rag.corpus_texts)
    rag.save_index("./rag_output/index")
    
    # 5️⃣ Train generator (optional)
    train_examples = [
        TrainExample(
            question=row["question"],
            answer=row["answer"]
        )
        for _, row in rag.df_train.head(1000).iterrows()
    ]
    rag.train_generator(train_examples, epochs=2)
    
    # 6️⃣ Test QA
    test_questions = [
        "Where is the Eiffel Tower?",
        "What is the capital of France?",
    ]
    
    for q in test_questions:
        answer = rag.answer(q, top_k=3)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

if __name__ == "__main__":
    main()
```

**Chạy:**
```bash
python full_pipeline.py
```

---

## 🔄 Training Details

### Retriever Training

**Objective:** Maximize similarity giữa question và positive passage

```python
rag.train_retriever(
    epochs=5,           # Number of epochs
    lr=2e-5,           # Learning rate
    patience=3,        # Early stopping patience
    temperature=20.0,  # Scaling factor for logits
    accumulation_steps=2,  # Gradient accumulation
    use_fp16=True,     # Mixed precision
    save_name="retriever_trained"
)
```

**Metrics:**
- `top_k_acc`: % gold passage trong top-k
- `mrr`: Mean Reciprocal Rank

### Generator Training

**Objective:** Predict answer tokens given (question + context)

```python
rag.train_generator(
    train_examples,
    batch_size=4,
    epochs=3,
    lr=5e-5,
    warmup_ratio=0.1,           # Warmup steps
    gradient_accumulation=1,    # Accumulation steps
    max_input_tokens=512,       # Max input length
    max_target_tokens=128,      # Max output length
    top_k=5,                    # Retrieve top-5 contexts
    save_name="generator_trained"
)
```

---

## 📊 Expected Performance

### On HotpotQA Dataset

| Component | Metric | Expected |
|-----------|--------|----------|
| **Retriever** | Top-5 Recall | 75-85% |
| | MRR | 0.55-0.65 |
| **Generator** | Exact Match | 30-40% |
| | BLEU-4 | 20-30 |
| **Full System** | Answer Correctness | 25-35% |

> Mục tiêu: Tăng độ chính xác bằng fine-tuning + ensemble

---

## 🔍 Model Selection Guide

### Retriever Models

```python
# Nhỏ, nhanh (58MB)
"sentence-transformers/all-MiniLM-L6-v2"  # Default, 384-dim

# Lớn, chính xác hơn (150MB)
"sentence-transformers/all-mpnet-base-v2"  # 768-dim

# Specialized for semantic search
"sentence-transformers/msmarco-distilbert-base-v4"
```

### Generator Models

```python
# Nhỏ, nhanh (300M params)
"google/flan-t5-small"

# Balanced (250M params) - DEFAULT
"google/flan-t5-base"

# Lớn, tốt hơn (780M params)
"google/flan-t5-large"

# Alternatives:
"meta-llama/Llama-2-7b"           # LLaMA
"mistralai/Mistral-7B"            # Mistral
```

---

## ⚡ Performance Tips

### Speed Up Training

```python
# 1. Batch size
rag.train_retriever(... , batch_size=32)  # Lớn hơn nếu GPU memory cho phép

# 2. Gradient accumulation (nếu batch size nhỏ)
rag.train_generator(..., gradient_accumulation=4)

# 3. Mixed precision (FP16)
rag.train_retriever(..., use_fp16=True)

# 4. Reduce sequence length
rag.train_generator(..., max_input_tokens=256)
```

### Reduce Memory Usage

```python
# 1. Smaller models
encoder="sentence-transformers/all-MiniLM-L6-v2"
generator="google/flan-t5-small"

# 2. Smaller batch size
rag.train_retriever(..., batch_size=8)

# 3. CPU mode
device="cpu"

# 4. Quantization (advanced)
# Requires bitsandbytes: pip install bitsandbytes
```

### Improve Quality

```python
# 1. Larger models
encoder="sentence-transformers/all-mpnet-base-v2"
generator="google/flan-t5-large"

# 2. Longer training
rag.train_retriever(..., epochs=10)

# 3. Lower learning rate
rag.train_retriever(..., lr=1e-5)

# 4. More negatives in retriever
# (modify create_retriever_train_loader)
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```python
# Giảm batch size
rag.train_retriever(..., batch_size=8)

# Hoặc dùng CPU
rag = RAGSystem(device="cpu")

# Hoặc clear cache
import torch
torch.cuda.empty_cache()
```

### Model Download Failed

```python
# 1. Kiểm tra HF token
echo $HF_TOKEN  # Linux/Mac
echo %HF_TOKEN% # Windows

# 2. Retry với token
os.environ["HF_TOKEN"] = "hf_..."
rag = RAGSystem(hf_token="hf_...")

# 3. Manual download
from huggingface_hub import snapshot_download
snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
```

### Slow Inference

```python
# Lớn batch size cho inference
embeddings = rag.encoder.encode(texts, batch_size=256)

# Hoặc dùng GPU
rag = RAGSystem(device="cuda")

# Hoặc smaller models
rag = RAGSystem(encoder_model="sentence-transformers/all-MiniLM-L6-v2")
```

---

## 📖 Documentation

- **ARCHITECTURE.md**: Chi tiết kiến trúc hệ thống
- **notebooks/rag.py**: Source code đầy đủ
- **notebooks/rag-hotpotqa.ipynb**: Interactive demo
- **HotpotQA**: https://hotpotqa.github.io/
- **Sentence Transformers**: https://www.sbert.net/
- **FAISS**: https://faiss.ai/

---

## 💡 Next Steps

1. ✅ Download HotpotQA data
2. ✅ Install dependencies: `pip install -r requirements.txt`
3. ✅ Run quick start example (Ví dụ 1)
4. ✅ Try full training pipeline (Ví dụ 2)
5. ✅ Evaluate retriever & generator
6. ✅ Fine-tune hyperparameters
7. ✅ Deploy as API (FastAPI)

---

## 📝 Citation

Nếu sử dụng repo này, vui lòng cite:

```bibtex
@dataset{hotpotqa,
  title={HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  year={2018}
}

@software{sentence_transformers,
  title={Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  year={2019}
}
```

---

## 📄 License

MIT License - Tự do dùng cho mục đích học tập & nghiên cứu

---

## 👨‍💻 Contributors

Made with ❤️ for HotpotQA RAG Research

# Answer questions
answer = rag.answer("Where is the Eiffel Tower?", top_k=2)
print(answer)
```

### Data Preparation: HotpotQA Format

Your CSV files should have columns:
- `question`: The question text
- `context`: JSON string with `{"title": [...], "sentences": [[...], [...]]}`
- `supporting_facts`: JSON string with `{"title": [...]}`
- `answer`: The answer text (for generator training)

Example:
```csv
question,context,supporting_facts,answer
"Who is the author of X?","{""title"": [""Person A"", ""Book X""], ...}","{""title"": [""Person A""]}","Person A wrote Book X"
```

### Workflow 1: Train Retriever on HotpotQA

```python
from rag import RAGSystem

rag = RAGSystem(
    output_dir="./rag_models",
    hf_token=os.getenv("HF_TOKEN")  # Optional
)

# Load data
rag.load_hotpotqa_data("train.csv", "valid.csv")
rag.build_corpus()

# Create training pairs
rag.create_retriever_train_loader(batch_size=16)

# Train
rag.train_retriever(
    epochs=5,
    lr=2e-5,
    patience=3,
    save_name="retriever_trained"
)

# Evaluate
metrics = rag.evaluate_retriever(top_k=5)
print(f"Top-5 Accuracy: {metrics['top5_acc']:.4f}")
print(f"MRR: {metrics['mrr']:.4f}")

# Build index for QA
rag.index_corpus(rag.corpus_texts)
rag.save_index()
```

### Workflow 2: Train Generator on HotpotQA

```python
from rag import RAGSystem, TrainExample

rag = RAGSystem(
    generator_model="google/flan-t5-base",
    output_dir="./rag_models"
)

# Load data
rag.load_hotpotqa_data("train.csv", "valid.csv")
rag.build_corpus()

# Build index first (for retrieval during training)
rag.index_corpus(rag.corpus_texts)

# Prepare training examples
train_examples = [
    TrainExample(
        question=row["question"],
        answer=row["answer"]
    )
    for _, row in rag.df_train.iterrows()
]

# Train generator
rag.train_generator(
    train_examples=train_examples,
    batch_size=4,
    epochs=3,
    lr=5e-5,
    top_k=5  # Use top-5 retrieved docs
)
```

### Workflow 3: Full RAG Pipeline

```python
from rag import RAGSystem, TrainExample

rag = RAGSystem(
    encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    generator_model="google/flan-t5-base",
    output_dir="./rag_models"
)

# 1. Load data
rag.load_hotpotqa_data("train.csv", "valid.csv")
rag.build_corpus()

# 2. Train retriever
rag.create_retriever_train_loader(batch_size=16)
rag.train_retriever(epochs=5, lr=2e-5)

# 3. Build index
rag.index_corpus(rag.corpus_texts)
rag.save_index()

# 4. Train generator
train_examples = [
    TrainExample(question=row["question"], answer=row["answer"])
    for _, row in rag.df_train.iterrows()
]
rag.train_generator(train_examples, epochs=3, batch_size=4)

# 5. Answer questions
answer = rag.answer("What is the capital of France?", top_k=5)
print(answer)
```

## Class Methods

### Data Loading
- `load_hotpotqa_data(train_path, valid_path)` - Load HotpotQA CSVs
- `build_corpus(df)` - Flatten docs into corpus texts
- `create_retriever_train_loader(batch_size)` - Create training pairs

### Retriever Training
- `train_retriever(epochs, lr, patience, temperature, ...)` - Train encoder
- `evaluate_retriever(top_k, rebuild_cache, ...)` - Eval on validation set

### Indexing
- `index_corpus(documents, batch_size)` - Build FAISS index
- `save_index(index_dir)` - Save index to disk
- `load_index(index_dir)` - Load index from disk

### Generator Training
- `train_generator(train_examples, epochs, lr, ...)` - Train generator

### QA
- `retrieve(query, top_k)` - Retrieve top-k documents
- `answer(question, top_k, max_new_tokens, temperature)` - Full QA pipeline

### Utilities
- `_build_prompt(question, contexts)` - Format prompt for generator

## Configuration

### Models

**Retriever (Encoder):**
- `sentence-transformers/all-MiniLM-L6-v2` (default, 22M params)
- `sentence-transformers/all-mpnet-base-v2` (109M params, better quality)
- `intfloat/e5-base-v2` (110M params, good for retrieval)
- `BAAI/bge-small-en-v1.5` (33M params, optimized for retrieval)

**Generator:**
- `google/flan-t5-base` (default, 250M params)
- `google/flan-t5-large` (770M params)
- `meta-llama/Llama-2-7b-hf` (requires token, 7B params)

### Hyperparameters

**Retriever Training:**
```python
rag.train_retriever(
    epochs=20,              # Number of epochs
    lr=2e-5,               # Learning rate
    patience=3,            # Early stopping patience
    temperature=20.0,      # Similarity scaling
    accumulation_steps=2,  # Gradient accumulation
    use_fp16=True,        # Mixed precision
)
```

**Generator Training:**
```python
rag.train_generator(
    batch_size=4,                 # Batch size
    epochs=3,                     # Epochs
    lr=5e-5,                      # Learning rate
    warmup_ratio=0.1,            # Warmup steps
    gradient_accumulation=1,      # Gradient accumulation
    max_input_tokens=512,         # Input max length
    max_target_tokens=128,        # Target max length
    top_k=5,                      # Retrieve top-k docs
)
```

## Environment Variables

```bash
# HuggingFace API token (for private models)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Disable offline mode (default: enabled)
HF_HUB_OFFLINE=0
```

## Output Structure

```
rag_output/
├── index/                       # FAISS index
│   ├── index.faiss
│   └── meta.json
├── corpus_embeds.pt           # Cached corpus embeddings
└── ...

models/                        # Cached models
├── sentence-transformers--all-MiniLM-L6-v2/
├── google--flan-t5-base/
└── retriever_trained/            # Fine-tuned retriever
└── generator_trained/            # Fine-tuned generator

```

## Examples

See [rag_example.py](rag_example.py) for complete examples:

```bash
# Basic QA (always works)
python rag_example.py

# Train retriever
python -c "from rag_example import example_retriever_training; example_retriever_training()"

# Train generator
python -c "from rag_example import example_generator_training; example_generator_training()"

# Full pipeline
python -c "from rag_example import example_end_to_end; example_end_to_end()"
```

## Troubleshooting

### Out of Memory (OOM)

**For Retriever:**
```python
rag.train_retriever(
    accumulation_steps=4,  # Increase gradient accumulation
    use_fp16=True,        # Enable mixed precision
)
# Or use smaller model:
rag.encoder = SentenceTransformer("all-MiniLM-L6-v2")
```

**For Generator:**
```python
rag.train_generator(
    batch_size=2,                # Reduce batch size
    gradient_accumulation=2,    # Increase accumulation
)
# Or use smaller model:
rag.generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
```

### Missing HuggingFace Token

If you get authentication errors:
1. Create token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Set `HF_TOKEN` environment variable
3. Or pass `hf_token` to RAGSystem:
   ```python
   rag = RAGSystem(hf_token="hf_xxxxxxxxxxxxxxxxxxxx")
   ```

### Data Format Errors

Ensure CSV has correct JSON format for `context` and `supporting_facts`:

```python
import json

# Correct format
context = {"title": ["Title1", "Title2"], "sentences": [["sent1", "sent2"], ["sent3"]]}
print(json.dumps(context))  # Verify this is valid JSON

# In CSV
df['context'] = df['context'].apply(json.dumps)  # If dict
```

## Performance Tips

1. **Use cached embeddings**: Set `rebuild_cache=False` for faster evaluation
2. **Gradient accumulation**: Simulate larger batches with gradient accumulation
3. **Mixed precision**: Use `use_fp16=True` for 2x speedup
4. **Smaller models**: Start with MiniLM for quick iteration
5. **Pre-compute contexts**: Pass `contexts` to TrainExample for faster training

## Citation

```bibtex
@article{lewis2020retrieval,
  title={Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and others},
  journal={NeurIPS},
  year={2020}
}

@inproceedings{yang2018hotpotqa,
  title={HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Shuai and others},
  booktitle={EMNLP},
  year={2018}
}
```

## License

MIT License
