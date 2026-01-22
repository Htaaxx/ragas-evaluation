# RAG System Architecture

## 📋 Tổng Quan

Hệ thống RAG (Retrieval-Augmented Generation) là một pipeline hoàn chỉnh kết hợp:
- **Retriever**: Mô hình embedding để tìm kiếm tài liệu liên quan
- **Index**: FAISS vector index cho tìm kiếm nhanh
- **Generator**: Mô hình seq2seq để tạo câu trả lời
- **QA Pipeline**: Lấy tài liệu → Sinh câu trả lời

---

## 🏗️ Cấu Trúc Dự Án

```
RAG with HotpotQA/
├── notebooks/                    # Jupyter notebooks & scripts
│   ├── rag.py                   # RAGSystem class - core logic
│   ├── rag-hotpotqa.ipynb       # Training pipeline on HotpotQA
│   ├── rag_example.py           # Usage examples
│   ├── data_collection.ipynb    # Data preprocessing
│   └── verbalization.ipynb      # Text processing utilities
│
├── data/                         # Datasets
│   └── hotpot_qa/
│       ├── train.csv            # Training set (train_size samples)
│       ├── valid.csv            # Validation set (valid_size samples)
│       ├── valid_async.csv      # Async validation variant
│       └── valid_async_checkpoint.csv  # Checkpoint
│
├── models/                       # Cached models (auto-downloaded)
│   ├── sentence-transformers--all-MiniLM-L6-v2/
│   ├── sentence-transformers--all-mpnet-base-v2/
│   └── google--flan-t5-base/
│
├── rag_output/                   # Outputs (auto-created)
│   ├── index/
│   │   ├── index.faiss          # FAISS index
│   │   └── meta.json            # Document metadata
│   └── *.pt                      # Embeddings cache
│
├── requirements.txt              # Python dependencies
├── README.md                     # Full documentation
├── ARCHITECTURE.md              # This file
└── .env                         # Environment variables (HF_TOKEN)
```

---

## 🔄 Workflow Pipeline

### 1. **Khởi Tạo Hệ Thống**

```
┌─────────────────────────────────────┐
│  RAGSystem.__init__()               │
├─────────────────────────────────────┤
│ 1. Load Encoder (Retriever)         │
│    - Download from HF hoặc cache    │
│    - Model: sentence-transformers   │
│                                     │
│ 2. Load Generator                   │
│    - Download from HF hoặc cache    │
│    - Model: T5/FLAN-T5             │
│                                     │
│ 3. Setup FAISS Index (empty)        │
│    - Ready cho indexing             │
│                                     │
│ 4. Create Output Directories        │
│    - models/, rag_output/          │
└─────────────────────────────────────┘
```

**Code:**
```python
rag = RAGSystem(
    encoder_model="sentence-transformers/all-MiniLM-L6-v2", # Retriever
    generator_model="google/flan-t5-base", # Generator
    models_dir="../models",      # Cache models ở đây
    output_dir="./rag_output",   # Outputs ở đây
    device="cuda"                # GPU hoặc CPU
)
```

---

### 2. **Tải & Xử Lý Dữ Liệu**

```
┌─────────────────────────────────────┐
│  load_hotpotqa_data()               │
├─────────────────────────────────────┤
│ CSV Input: question | answer | ...  │
│           context (JSON string)     │
│                                     │
│ Parse context: tách title & text    │
│                                     │
│ Output: df_train, df_valid          │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  build_corpus()                     │
├─────────────────────────────────────┤
│ Flatten tất cả docs từ DataFrame    │
│                                     │
│ Output:                             │
│  - corpus_texts: list[str]          │
│  - doc_titles: list[str]            │
└─────────────────────────────────────┘
```

**Code:**
```python
# Tải dữ liệu HotpotQA
rag.load_hotpotqa_data(
    train_path="data/hotpot_qa/train.csv",
    valid_path="data/hotpot_qa/valid.csv"
)

# Build corpus (tất cả documents)
corpus_texts, doc_titles = rag.build_corpus()
```

---

### 3. **Training Retriever**

```
┌──────────────────────────────────────────┐
│  create_retriever_train_loader()         │
├──────────────────────────────────────────┤
│ Tạo training pairs:                      │
│  (question, positive_passage)            │
│                                          │
│ Từ HotpotQA:                             │
│  - Gold passages: supporting_facts       │
│  - Negative: các passages khác           │
│                                          │
│ Output: DataLoader[InputExample]         │
└──────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│  train_retriever()                       │
├──────────────────────────────────────────┤
│ For each epoch:                          │
│  1. Encode questions → q_emb             │
│  2. Encode passages → p_emb              │
│  3. Compute similarity: sim = q_emb × p_emb^T
│  4. Cross-entropy loss: wrong pair → high loss
│  5. Backprop & optimize                  │
│                                          │
│ Early stopping khi loss không giảm      │
│                                          │
│ Output: trained encoder model            │
└──────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│  evaluate_retriever()                    │
├──────────────────────────────────────────┤
│ Metrics:                                 │
│  - Top-K Accuracy: gold passage in top-k │
│  - MRR: Mean Reciprocal Rank            │
│                                          │
│ Output: {top1_acc, top3_acc, top5_acc,   │
│          mrr, ...}                       │
└──────────────────────────────────────────┘
```

**Code:**
```python
# Tạo training data
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
```

---

### 4. **Indexing Documents**

```
┌──────────────────────────────────────────┐
│  index_corpus(documents)                 │
├──────────────────────────────────────────┤
│ 1. Encode tất cả documents:              │
│    embs = encoder.encode(documents)      │
│    Shape: (num_docs, embedding_dim)      │
│                                          │
│ 2. Normalize L2 embeddings               │
│                                          │
│ 3. Tạo FAISS IndexFlatIP:                │
│    - Supports inner product search       │
│    - Index.add(embeddings)               │
│                                          │
│ 4. Store docstore: list[str]             │
│                                          │
│ Output: faiss.IndexFlatIP                │
└──────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│  save_index(index_dir)                   │
├──────────────────────────────────────────┤
│ Lưu:                                     │
│  - index.faiss: Binary FAISS index       │
│  - meta.json: {docstore, doc_titles}    │
│                                          │
│ Có thể load lại sau với load_index()    │
└──────────────────────────────────────────┘
```

**Code:**
```python
# Index corpus
rag.index_corpus(rag.corpus_texts, batch_size=64)

# Save cho lần sau
rag.save_index("./rag_output/index")
```

---

### 5. **Training Generator**

```
┌───────────────────────────────────────────┐
│  train_generator(train_examples)          │
├───────────────────────────────────────────┤
│ For each training example:                │
│                                           │
│ 1. Retrieve top-K passages:               │
│    contexts = rag.retrieve(question)      │
│                                           │
│ 2. Build prompt:                          │
│    prompt = "Context:\n{ctx}\nQ: ..."    │
│                                           │
│ 3. Encode input (max_input_tokens=512):  │
│    input_ids = tokenizer(prompt)          │
│                                           │
│ 4. Encode output (max_target_tokens=128):│
│    labels = tokenizer(answer)             │
│                                           │
│ 5. Forward pass:                          │
│    output = generator(input_ids, labels=labels)
│                                           │
│ 6. Backward & optimize:                   │
│    loss.backward()                        │
│    optimizer.step()                       │
│                                           │
│ Output: trained generator + tokenizer     │
└───────────────────────────────────────────┘
```

**Code:**
```python
train_examples = [
    TrainExample(
        question="Where is the Eiffel Tower?",
        answer="Paris, France",
        contexts=None  # Auto-retrieve
    ),
    ...
]

rag.train_generator(
    train_examples,
    batch_size=4,
    epochs=3,
    lr=5e-5,
    save_name="generator_trained"
)
```

---

### 6. **Question Answering Pipeline**

```
┌────────────────────────────────────────┐
│  answer(question)                      │
├────────────────────────────────────────┤
│                                        │
│ 1. RETRIEVAL:                          │
│    q_emb = encoder.encode(question)    │
│    top_k_indices = faiss.search(q_emb)│
│    contexts = [docstore[i] for i ...]  │
│                                        │
│ 2. PROMPT BUILDING:                    │
│    prompt = "Use context to answer:   │
│    Context: {ctx1}\n{ctx2}\n...       │
│    Question: {question}\nAnswer:"      │
│                                        │
│ 3. GENERATION:                         │
│    input_ids = tokenizer(prompt)      │
│    output_ids = generator.generate(    │
│        input_ids,                      │
│        max_new_tokens=64               │
│    )                                   │
│    answer = tokenizer.decode(output)   │
│                                        │
│ Output: str (generated answer)         │
└────────────────────────────────────────┘
```

**Code:**
```python
# Trả lời câu hỏi
answer = rag.answer(
    question="What is the capital of France?",
    top_k=5,
    max_new_tokens=64,
    temperature=0.7  # For sampling
)
print(answer)  # "Paris"
```

---

## 🧠 Core Components

### 1. **Retriever (Encoder)**
- **Mô hình**: `sentence-transformers/all-MiniLM-L6-v2` (default)
- **Chức năng**: Đưa text → vector embeddings
- **Kích thước**: 384 chiều
- **Use case**: Tìm documents liên quan đến query

**Alternatives:**
```python
# Lớn hơn, tốt hơn (đó 150MB)
"sentence-transformers/all-mpnet-base-v2"  # 768-dim

# Nhỏ, nhanh (58MB)
"sentence-transformers/all-MiniLM-L6-v2"   # 384-dim (default)
```

---

### 2. **Generator**
- **Mô hình**: `google/flan-t5-base` (default)
- **Kiến trúc**: Seq2Seq transformer
- **Chức năng**: text → text (question + context → answer)
- **Kích thước**: ~900MB

**Alternatives:**
```python
# Nhỏ, nhanh
"google/flan-t5-small"   # 300M params

# Default
"google/flan-t5-base"    # 250M params

# Lớn, tốt hơn
"google/flan-t5-large"   # 780M params
```

---

### 3. **FAISS Index**
- **Loại**: `IndexFlatIP` (Inner Product)
- **Độ phức tạp**: O(n) search, không nén
- **Ưu điểm**: Chính xác 100%, đơn giản
- **Nhược điểm**: Chậm với corpus lớn (>1M)

**Alternatives:**
```python
# Nén, nhanh hơn
faiss.IndexIVFFlat(quantizer, nlist=100)

# GPU-based
faiss.GpuIndexFlatIP(res, d)
```

---

### 4. **Data Structures**

#### `TrainExample`
```python
@dataclass
class TrainExample:
    question: str
    answer: str
    contexts: Sequence[str] | None = None  # Pre-retrieved hoặc None
```

#### `RetrieverExample`
```python
@dataclass
class RetrieverExample:
    question: str
    positive_passage: str
    negative_passages: Sequence[str] | None = None
```

---

## 💾 Model Caching System

### Tự động Caching

```
1. User request: encoder_model="sentence-transformers/all-MiniLM-L6-v2"

2. Check local: ../models/sentence-transformers--all-MiniLM-L6-v2/
   
3a. Tìm thấy → Load từ cache (nhanh! ✅)
   
3b. Không tìm → Download từ HuggingFace (đầu tiên)
                 → Move tới cache
                 → Load từ cache lần sau
```

### Lợi ích

✅ Tránh tải lại model mỗi lần khởi động  
✅ Offline mode (sau khi cached)  
✅ Tập trung quản lý models  
✅ Dễ thay đổi model  

### Ví dụ

```python
# Lần 1: Download (slow)
rag1 = RAGSystem(encoder_model="sentence-transformers/all-MiniLM-L6-v2")
# ✅ Loaded from HF, saved to ../models/

# Lần 2+: Load từ cache (fast)
rag2 = RAGSystem(encoder_model="sentence-transformers/all-MiniLM-L6-v2")
# ✅ Loaded from cache: sentence-transformers--all-MiniLM-L6-v2
```

---

## 📊 Data Format

### HotpotQA CSV Format

```csv
question,answer,context,supporting_facts
"Who founded Microsoft?","Bill Gates and Paul Allen","{""title"":[""Bill Gates"",""Paul Allen""],""sentences"":[[""Sentence 1"",""Sentence 2""],[...]]}","{""title"":[""Bill Gates""]}"
```

**Parsing:**
```python
rag.load_hotpotqa_data(train_path, valid_path)
# → df["docs"] = [{"title": "...", "text": "..."}, ...]
# → df["supporting_facts"]["title"] = ["Gold Doc 1", "Gold Doc 2"]
```

---

## 🔍 Retrieval Process

### Inner Product Search

```
1. Query: "What is AI?"
   ↓
2. Encode query:
   q_emb = encoder.encode("What is AI?")
   # q_emb: shape (384,)
   
3. FAISS search (IndexFlatIP):
   scores, idxs = index.search(q_emb, k=5)
   # scores: inner product similarities
   # idxs: top-5 document indices
   
4. Retrieve documents:
   contexts = [docstore[i] for i in idxs[0]]
   
5. Output: top_k most similar documents
```

### Similarity Metric

- **Inner Product** (sử dụng vì embeddings normalized L2)
- = Cosine Similarity (khi vectors đã normalize)
- Range: [0, 1] (sau normalize)

---

## 🎯 Training Strategies

### Retriever Training

**Loss Function:** CrossEntropyLoss (ranking loss)
```python
# Multi-GPU contrastive learning
similarities = Q_emb @ P_emb^T  # (batch_size, batch_size)
labels = torch.arange(batch_size)  # Diagonal positives
loss = CrossEntropyLoss(similarities, labels)
```

**Temperature Scaling:**
```python
sim = sim * temperature  # Scale before softmax
# Cao → softer distribution
# Thấp → harder distribution (default 20.0)
```

### Generator Training

**Seq2Seq Loss:**
```python
# Input: prompt (question + context)
# Target: answer
# Loss: Cross-entropy on token prediction

loss = generator(input_ids, labels=target_ids).loss
```

---

## ⚡ Performance Optimization

### Batch Processing

```python
# Encoder batch processing
embeddings = encoder.encode(
    texts,
    batch_size=64,          # Process 64 texts at a time
    convert_to_tensor=True
)
```

### Gradient Accumulation

```python
# Khi batch_size quá nhỏ cho GPU:
for step, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision (FP16)

```python
# Accelerate training trên GPU:
with autocast(enabled=True):  # Automatic FP16
    output = model(input)
    loss = criterion(output, target)
```

---

## 🛠️ Configuration Examples

### Quick Start (CPU)
```python
rag = RAGSystem(device="cpu")
```

### Production (GPU, large models)
```python
rag = RAGSystem(
    encoder_model="sentence-transformers/all-mpnet-base-v2",
    generator_model="google/flan-t5-large",
    device="cuda",
    models_dir="/shared/models",  # NFS drive
)
```

### Research (multiple models, fast caching)
```python
for encoder in [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]:
    rag = RAGSystem(encoder_model=encoder, models_dir="../models")
    # Models share cache automatically
```

---

## 📈 Typical Metrics

### Retriever Evaluation (HotpotQA)

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 45-55% |
| Top-3 Accuracy | 65-75% |
| Top-5 Accuracy | 75-85% |
| MRR | 0.55-0.65 |

### Generator Evaluation

- **BLEU-4**: 20-30
- **ROUGE-L**: 0.35-0.45
- **Exact Match**: 30-40%

---

## 🔗 External Resources

- [HotpotQA Dataset](https://hotpotqa.github.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [FLAN-T5 Paper](https://arxiv.org/abs/2301.13688)

---

## 📝 Notes

- Models tải từ HuggingFace Hub (cần kết nối internet lần đầu)
- Embeddings được normalize L2 cho inner product = cosine similarity
- FAISS IndexFlatIP không nén → chính xác 100% nhưng chậm với lớp dữ liệu lớn
- Có thể mở rộng sang GPU FAISS, quantization, hay hybrid search
