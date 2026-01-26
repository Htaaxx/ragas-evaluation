# Usage Guide - RAG Training System

## 🎯 What This System Does

This system allows you to **train your own RAG models** instead of using pre-trained ones. You can:

1. **Train a retriever** - Fine-tune sentence transformers to find relevant documents
2. **Train a generator** - Fine-tune T5/FLAN-T5 to generate answers with retrieved contexts
3. **Build indices** - Create FAISS indices for fast retrieval
4. **Answer questions** - Use your trained models for QA

## 🚀 Getting Started

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Data

Place HotpotQA CSV files in:
- `data/hotpot_qa/train.csv`
- `data/hotpot_qa/valid.csv`

### Step 3: Choose Your Training Mode

## 📋 Training Modes

### Mode 1: Train Retriever Only (Recommended for First Run)

This trains just the retriever model to find relevant documents.

**Command Line:**
```bash
python train.py --train-retriever-only \
    --train-data data/hotpot_qa/train.csv \
    --valid-data data/hotpot_qa/valid.csv \
    --retriever-epochs 5
```

**Python:**
```python
from src.config import RAGConfig
from src.rag_system import RAGSystem

config = RAGConfig(
    train_data_path="data/hotpot_qa/train.csv",
    valid_data_path="data/hotpot_qa/valid.csv",
    retriever_epochs=5
)

rag = RAGSystem(config=config)
rag.load_data()
rag.train_retriever()
rag.evaluate_retriever()
rag.build_index()
```

**What happens:**
1. Loads your data
2. Trains retriever with contrastive learning
3. Evaluates on validation set (Recall@K, MRR)
4. Builds FAISS index
5. Saves trained model to `../models/retriever_trained`

### Mode 2: Train Both Retriever and Generator

This trains both components for a complete RAG system.

**Command Line:**
```bash
python train.py --train-retriever --train-generator \
    --train-data data/hotpot_qa/train.csv \
    --valid-data data/hotpot_qa/valid.csv \
    --retriever-epochs 5 \
    --generator-epochs 3
```

**Python:**
```python
from src.config import RAGConfig
from src.rag_system import RAGSystem

config = RAGConfig(
    train_data_path="data/hotpot_qa/train.csv",
    valid_data_path="data/hotpot_qa/valid.csv",
    retriever_epochs=5,
    generator_epochs=3
)

rag = RAGSystem(config=config)
rag.load_data()
rag.train_retriever()
rag.evaluate_retriever()
rag.build_index()
rag.train_generator()
```

**What happens:**
1. Everything from Mode 1
2. Plus: Trains generator to produce answers
3. Saves generator to `../models/generator_trained`

### Mode 3: Use Pretrained Models (Inference Only)

Load your trained models and use them for QA.

```python
from src.config import RAGConfig
from src.rag_system import RAGSystem

config = RAGConfig()
rag = RAGSystem.from_pretrained(
    encoder_path="../models/retriever_trained",
    generator_path="../models/generator_trained",
    index_path="./rag_output/index",
    config=config
)

# Answer questions
answer = rag.answer("What is the capital of France?")
print(answer)

# Get contexts too
answer, contexts = rag.answer("Who founded Microsoft?", return_contexts=True)
print(f"Answer: {answer}")
print(f"Contexts: {contexts}")
```

## ⚙️ Configuration

### Basic Configuration

```python
from src.config import RAGConfig

config = RAGConfig(
    # Data paths
    train_data_path="data/hotpot_qa/train.csv",
    valid_data_path="data/hotpot_qa/valid.csv",
    
    # Training
    retriever_epochs=5,
    retriever_batch_size=16,
    generator_epochs=3,
    generator_batch_size=4,
    
    # Device
    device="cuda"  # or "cpu"
)
```

### Advanced Configuration

```python
config = RAGConfig(
    # Use different models
    encoder_model="sentence-transformers/all-mpnet-base-v2",  # Larger
    generator_model="google/flan-t5-large",  # Larger
    
    # Retriever training
    retriever_epochs=10,
    retriever_batch_size=32,
    retriever_lr=2e-5,
    retriever_patience=3,
    retriever_temperature=20.0,
    retriever_use_fp16=True,  # Mixed precision
    
    # Generator training
    generator_epochs=5,
    generator_batch_size=8,
    generator_lr=5e-5,
    generator_max_input_tokens=512,
    
    # Retrieval
    top_k=5,
    
    # Paths
    models_dir="/path/to/models",
    output_dir="/path/to/output"
)
```

## 📊 Understanding the Output

### During Training

You'll see:
```
🔍 Initializing RAG System
================================
Device: cuda
Encoder: sentence-transformers/all-MiniLM-L6-v2
Generator: google/flan-t5-base

📥 Loading models...
✅ Loaded from cache: sentence-transformers--all-MiniLM-L6-v2
✅ Loaded from cache: google--flan-t5-base

📂 Loading HotpotQA data...
✅ Loaded 1000 training samples
✅ Loaded 200 validation samples

🚀 Training Retriever...
Epoch 1/5: 100%|████████| 63/63 [00:45<00:00, loss=2.3456]
📉 Epoch 1/5: avg_loss=2.3456
✅ Saved best model → ../models/retriever_trained

📊 Evaluating Retriever...
recall@1    : 0.4523
recall@3    : 0.6734
recall@5    : 0.7845
mrr         : 0.5678
```

### After Training

Models are saved to:
- `../models/retriever_trained/` - Your trained retriever
- `../models/generator_trained/` - Your trained generator
- `./rag_output/index/` - FAISS index with metadata

## 🎓 Common Use Cases

### Use Case 1: Quick Test with Small Data

```python
config = RAGConfig(
    max_train_samples=100,  # Use only 100 samples
    max_valid_samples=20,
    retriever_epochs=2,
    generator_epochs=1
)

rag = RAGSystem(config=config)
rag.load_data()
rag.train_retriever()
rag.build_index()
```

### Use Case 2: Production Training

```python
config = RAGConfig(
    encoder_model="sentence-transformers/all-mpnet-base-v2",
    generator_model="google/flan-t5-large",
    retriever_epochs=10,
    generator_epochs=5,
    device="cuda",
    retriever_use_fp16=True
)

rag = RAGSystem(config=config)
rag.load_data()
rag.train_retriever()
rag.evaluate_retriever()
rag.build_index()
rag.train_generator()
```

### Use Case 3: Only Build Index (No Training)

```python
from src.rag_system import RAGSystem

rag = RAGSystem()

# Your documents
docs = [
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
    # ... more documents
]

# Build index
rag.indexer.build_index(documents=docs)
rag.indexer.save_index()

# Use for QA
answer = rag.answer("What is the capital of France?")
```

## 🔧 Troubleshooting

### Problem: CUDA Out of Memory

**Solution:** Reduce batch sizes
```python
config = RAGConfig(
    retriever_batch_size=8,   # Reduce from 16
    generator_batch_size=2,   # Reduce from 4
)
```

### Problem: Training Too Slow

**Solution:** Use smaller models or fewer epochs
```python
config = RAGConfig(
    encoder_model="sentence-transformers/all-MiniLM-L6-v2",  # Smaller
    generator_model="google/flan-t5-small",  # Smaller
    retriever_epochs=3,  # Fewer epochs
)
```

### Problem: Data Not Found

**Solution:** Check file paths
```bash
ls data/hotpot_qa/
# Should show: train.csv, valid.csv
```

### Problem: Poor Performance

**Solution:** Train longer or use larger models
```python
config = RAGConfig(
    retriever_epochs=10,  # More epochs
    encoder_model="sentence-transformers/all-mpnet-base-v2",  # Larger model
)
```

## 📖 Where to Go Next

1. **Run Examples:**
   ```bash
   python examples/basic_qa.py
   python examples/train_retriever.py
   python examples/end_to_end.py
   ```

2. **Read Documentation:**
   - `README.md` - Full documentation
   - `QUICKSTART.md` - Quick reference
   - `ARCHITECTURE.md` - System architecture

3. **Explore Code:**
   - `src/config.py` - All configuration options
   - `src/rag_system.py` - Main system class
   - `train.py` - Training script

## 💡 Tips

1. **Start small** - Use `max_train_samples` to test with small data first
2. **Monitor GPU** - Use `nvidia-smi` to check GPU usage
3. **Save often** - Models are automatically saved during training
4. **Use caching** - Models are cached in `../models/` to avoid re-downloads
5. **Check metrics** - Look at Recall@K and MRR to evaluate retriever

## 🎉 You're Ready!

Start with:
```bash
python train.py --train-retriever-only \
    --train-data data/hotpot_qa/train.csv \
    --valid-data data/hotpot_qa/valid.csv
```

Good luck with your RAG training! 🚀
