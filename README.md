# RAG Training System

A complete system for **training and fine-tuning your own RAG (Retrieval-Augmented Generation) models** from scratch, rather than using pre-trained models.

## 🎯 Overview

This project provides a comprehensive framework for:

- **Training custom retriever models** using contrastive learning on question-passage pairs
- **Fine-tuning generator models** (T5, FLAN-T5) with retrieved contexts
- **Building efficient FAISS indices** for fast similarity search
- **End-to-end question answering** with trained models
- **Automatic model caching** to avoid repeated downloads

### Key Difference from Traditional RAG

- **Traditional RAG**: Uses pre-trained models with retrieval (off-the-shelf)
- **This System**: Trains/fine-tunes models specifically for your RAG task

## 📋 Features

✅ **Retriever Training**
- Contrastive learning with in-batch negatives
- Support for sentence-transformers models
- Early stopping and model checkpointing
- Mixed precision training (FP16)

✅ **Generator Training**
- Seq2seq fine-tuning with retrieved contexts
- Support for T5/FLAN-T5 models
- Gradient accumulation for large batches
- Learning rate scheduling with warmup

✅ **Efficient Indexing**
- FAISS vector index for fast retrieval
- Batch encoding for large corpora
- Save/load functionality

✅ **Model Caching**
- Automatic HuggingFace model download
- Local caching to avoid re-downloads
- Organized storage structure

✅ **Evaluation**
- Recall@K, Precision@K, MRR metrics
- Corpus embedding caching
- Comprehensive evaluation reports

## 🏗️ Project Structure

```
ragas-evaluation/
├── src/                          # Source code
│   ├── config.py                # Configuration management
│   ├── rag_system.py            # Main RAG system class
│   ├── data/                    # Data loading
│   │   ├── __init__.py
│   │   └── loader.py            # HotpotQA data loader
│   ├── training/                # Training modules
│   │   ├── __init__.py
│   │   ├── retriever_trainer.py # Retriever training
│   │   └── generator_trainer.py # Generator training
│   ├── retrieval/               # Retrieval & QA
│   │   ├── __init__.py
│   │   ├── indexer.py           # FAISS indexing
│   │   └── qa_pipeline.py       # QA pipeline
│   ├── evaluation/              # Evaluation
│   │   ├── __init__.py
│   │   └── retriever_evaluator.py
│   └── utils/                   # Utilities
│       ├── __init__.py
│       └── model_cache.py       # Model caching
├── examples/                     # Example scripts
│   ├── basic_qa.py              # Basic QA without training
│   ├── train_retriever.py       # Retriever training example
│   └── end_to_end.py            # Full pipeline example
├── data/                         # Data directory
│   └── hotpot_qa/
│       ├── train.csv
│       └── valid.csv
├── train.py                      # Main training script
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── ARCHITECTURE.md              # Detailed architecture docs
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ragas-evaluation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

The system expects HotpotQA data in CSV format:

```csv
question,answer,context,supporting_facts
"Who founded Microsoft?","Bill Gates and Paul Allen","{""title"":[...],""sentences"":[...]}","{""title"":[...]}"
```

Place your data in:
- `data/hotpot_qa/train.csv`
- `data/hotpot_qa/valid.csv`

### 3. Train Your RAG System

#### Option A: Train Retriever Only

```bash
python train.py --train-retriever-only \
    --train-data data/hotpot_qa/train.csv \
    --valid-data data/hotpot_qa/valid.csv \
    --retriever-epochs 5 \
    --retriever-batch-size 16
```

#### Option B: Train Both Retriever and Generator

```bash
python train.py --train-retriever --train-generator \
    --train-data data/hotpot_qa/train.csv \
    --valid-data data/hotpot_qa/valid.csv \
    --retriever-epochs 5 \
    --generator-epochs 3
```

#### Option C: Use Configuration File

```bash
python train.py --config config.json --train-retriever --train-generator
```

### 4. Use Trained Models

```python
from src.config import RAGConfig
from src.rag_system import RAGSystem

# Load trained system
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
```

## 📚 Usage Examples

### Example 1: Basic QA (No Training)

```python
from src.config import RAGConfig
from src.rag_system import RAGSystem

# Initialize system
config = RAGConfig()
rag = RAGSystem(config=config)

# Create corpus
documents = [
    "The Eiffel Tower is in Paris, France.",
    "Paris is the capital of France.",
]

# Build index
rag.indexer.build_index(documents=documents)

# Answer questions
answer = rag.answer("Where is the Eiffel Tower?")
print(answer)  # "Paris, France"
```

See `examples/basic_qa.py` for full example.

### Example 2: Train Retriever

```python
from src.config import RAGConfig
from src.rag_system import RAGSystem

# Configure
config = RAGConfig(
    train_data_path="data/hotpot_qa/train.csv",
    valid_data_path="data/hotpot_qa/valid.csv",
    retriever_epochs=5
)

# Initialize and train
rag = RAGSystem(config=config)
rag.load_data()
rag.train_retriever()
rag.evaluate_retriever()
rag.build_index()
```

See `examples/train_retriever.py` for full example.

### Example 3: End-to-End Training

```python
from src.config import RAGConfig
from src.rag_system import RAGSystem

# Configure
config = RAGConfig(
    train_data_path="data/hotpot_qa/train.csv",
    valid_data_path="data/hotpot_qa/valid.csv"
)

# Full pipeline
rag = RAGSystem(config=config)
rag.load_data()
rag.train_retriever()
rag.evaluate_retriever()
rag.build_index()
rag.train_generator()

# Test QA
answer = rag.answer("Your question here")
```

See `examples/end_to_end.py` for full example.

## ⚙️ Configuration

The system is highly configurable through the `RAGConfig` class:

```python
from src.config import RAGConfig

config = RAGConfig(
    # Models
    encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    generator_model="google/flan-t5-base",
    
    # Paths
    models_dir="../models",
    output_dir="./rag_output",
    train_data_path="data/hotpot_qa/train.csv",
    valid_data_path="data/hotpot_qa/valid.csv",
    
    # Retriever training
    retriever_epochs=5,
    retriever_batch_size=16,
    retriever_lr=2e-5,
    retriever_patience=3,
    
    # Generator training
    generator_epochs=3,
    generator_batch_size=4,
    generator_lr=5e-5,
    
    # Retrieval
    top_k=5,
    
    # Device
    device="cuda"  # or "cpu"
)
```

See `src/config.py` for all available options.

## 📊 Training Details

### Retriever Training

- **Method**: Contrastive learning with in-batch negatives
- **Loss**: CrossEntropyLoss on similarity matrix
- **Features**:
  - Temperature scaling
  - Gradient accumulation
  - Mixed precision (FP16)
  - Early stopping
  - Model checkpointing

### Generator Training

- **Method**: Seq2seq fine-tuning with retrieved contexts
- **Loss**: Cross-entropy on token prediction
- **Features**:
  - Learning rate warmup
  - Gradient accumulation
  - Dynamic context retrieval
  - Model checkpointing

### Evaluation Metrics

- **Recall@K**: Percentage of queries with relevant doc in top-K
- **Precision@K**: Average fraction of relevant docs in top-K
- **MRR**: Mean Reciprocal Rank of first relevant document

## 🔧 Advanced Usage

### Custom Models

```python
config = RAGConfig(
    encoder_model="sentence-transformers/all-mpnet-base-v2",  # Larger encoder
    generator_model="google/flan-t5-large",  # Larger generator
)
```

### GPU Training

```python
config = RAGConfig(
    device="cuda",
    retriever_batch_size=32,  # Larger batch for GPU
    retriever_use_fp16=True,  # Mixed precision
)
```

### Custom Data

```python
from src.data.loader import HotpotQALoader

loader = HotpotQALoader()
loader.load_data("my_train.csv", "my_valid.csv")

# Use with RAG system
rag.data_loader = loader
```

## 📖 Documentation

- **README.md** (this file): Quick start and usage
- **ARCHITECTURE.md**: Detailed system architecture
- **src/config.py**: Configuration options
- **examples/**: Working code examples

## 🛠️ Development

### Running Examples

```bash
# Basic QA
python examples/basic_qa.py

# Train retriever
python examples/train_retriever.py

# End-to-end training
python examples/end_to_end.py
```

### Project Structure

The codebase is organized into modules:

- **config**: Configuration management
- **data**: Data loading and preprocessing
- **training**: Training logic for retriever and generator
- **retrieval**: Indexing and QA pipeline
- **evaluation**: Evaluation metrics
- **utils**: Utilities (model caching, etc.)

## 📝 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Sentence Transformers 2.2+
- FAISS (CPU or GPU)
- See `requirements.txt` for full list

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- HuggingFace for Transformers and Sentence Transformers
- Facebook AI for FAISS
- HotpotQA dataset creators

## 📧 Contact

[Your Contact Information]

---

**Note**: This system is designed for training RAG models from scratch. For production use with pre-trained models, consider using LangChain or similar frameworks.
