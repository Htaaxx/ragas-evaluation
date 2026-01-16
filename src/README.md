# RAG System - Modular Architecture

This directory contains a modular RAG (Retrieval-Augmented Generation) system for the HotPotQA dataset.

## 📁 Project Structure

```
src/
├── config.py                  # Configuration settings and constants
├── rag_system.py             # Main RAG system orchestrator
├── run_rag.py                # Command-line interface
│
├── models/                   # LLM wrappers
│   ├── __init__.py
│   ├── huggingface_llm.py   # HuggingFace Inference API wrapper
│   └── gemini_llm.py        # Google Gemini wrapper
│
├── data/                     # Data loading and processing
│   ├── __init__.py
│   └── loader.py            # HotPotQA dataset loader
│
├── vectorstore/             # Vector store management
│   ├── __init__.py
│   └── manager.py           # FAISS vector store manager
│
├── evaluation/              # Evaluation metrics
│   ├── __init__.py
│   └── ragas_evaluator.py  # RAGAS evaluation framework
│
├── utils/                   # Utility functions
│   ├── __init__.py
│   └── file_utils.py       # File I/O utilities
│
└── basic_rag.py            # DEPRECATED - kept for backward compatibility
```

## 🎯 Key Components

### 1. **Configuration (`config.py`)**
Central configuration for all system settings:
- Model names and parameters
- Chunking and retrieval settings
- API configurations
- Prompt templates

### 2. **RAG System (`rag_system.py`)**
Main orchestrator class that coordinates:
- Data loading
- Vector store creation/loading
- QA chain setup
- Query processing
- RAGAS evaluation

### 3. **Models Package (`models/`)**
LLM wrappers for different providers:
- **HuggingFace**: Direct HTTP API calls for compatibility
- **Gemini**: Google Gemini API wrapper

### 4. **Data Package (`data/`)**
Dataset loading and processing:
- Load from HuggingFace Hub
- Load from local JSON files
- Document preprocessing

### 5. **Vector Store Package (`vectorstore/`)**
Vector store management:
- Document chunking
- Embedding generation
- FAISS index creation/loading
- Retriever configuration

### 6. **Evaluation Package (`evaluation/`)**
RAGAS evaluation framework:
- Multiple metrics support
- Flexible LLM configuration
- Result processing

### 7. **Utils Package (`utils/`)**
Helper functions:
- JSON file I/O
- Common utilities

## 🚀 Usage

### Build Vector Store
```bash
python src/run_rag.py --mode build --max-samples 1000
```

### Interactive Query
```bash
# Using HuggingFace
python src/run_rag.py --mode query

# Using Google Gemini (recommended)
python src/run_rag.py --mode query --use-gemini
```

### Batch Evaluation
```bash
python src/run_rag.py --mode evaluate --num-questions 10 --use-gemini
```

### RAGAS Evaluation
```bash
python src/run_rag.py --mode ragas --num-questions 5 --use-gemini
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:
```env
# HuggingFace API Token (for HuggingFace models)
HUGGINGFACEHUB_API_TOKEN=your_token_here

# Google API Key (for Gemini models)
GOOGLE_API_KEY=your_key_here

# OpenAI API Key (optional, for RAGAS evaluation)
OPENAI_API_KEY=your_key_here
```

### Custom Configuration
Edit `src/config.py` to change:
- Default models
- Chunking parameters
- Retrieval settings
- Temperature and generation parameters

## 📦 Dependencies

Key dependencies:
- `langchain` - LLM framework
- `langchain-google-genai` - Google Gemini integration
- `ragas` - RAG evaluation framework
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Embeddings
- `datasets` - HuggingFace datasets

See `requirements.txt` for complete list.

## 🎨 Design Principles

### 1. **Separation of Concerns**
Each module has a single, well-defined responsibility:
- Models handle LLM interactions
- Data handles dataset loading
- Vectorstore handles retrieval
- Evaluation handles metrics

### 2. **Dependency Injection**
Components are loosely coupled and can be easily swapped:
```python
# Easy to switch between LLM providers
rag.setup_qa_chain(use_gemini=True)  # or False for HuggingFace
```

### 3. **Configuration Management**
All settings centralized in `config.py`:
- Easy to modify
- Type-safe defaults
- Environment variable support

### 4. **Extensibility**
Easy to add new components:
- New LLM providers → Add to `models/`
- New metrics → Add to `evaluation/`
- New data sources → Add to `data/`

## 🔄 Migration from Old Code

If you were using `basic_rag.py`:

**Old:**
```python
from basic_rag import BasicRAG

rag = BasicRAG()
```

**New:**
```python
from rag_system import RAGSystem

rag = RAGSystem()
```

The API is largely compatible, with the same method names and signatures.

## 🧪 Testing

Test individual components:

```python
# Test data loading
from data.loader import DataLoader
docs, questions = DataLoader.load_hotpotqa(max_samples=10)

# Test vector store
from vectorstore.manager import VectorStoreManager
vsm = VectorStoreManager()
vsm.create_vectorstore(docs)

# Test LLM
from models.gemini_llm import create_gemini_llm
llm = create_gemini_llm()
```

## 📝 Best Practices

1. **Use configuration constants** instead of hardcoded values
2. **Import from package level** (`from models import HuggingFaceInferenceLLM`)
3. **Handle errors gracefully** with try-except blocks
4. **Log important operations** for debugging
5. **Use type hints** for better IDE support

## 🐛 Troubleshooting

### Import Errors
Make sure you're in the project root and have activated the virtual environment:
```bash
cd /path/to/ragas-evaluation
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac
```

### API Key Errors
Check your `.env` file and ensure API keys are set correctly.

### Module Not Found
Ensure all `__init__.py` files are present in package directories.

## 📚 Further Reading

- [LangChain Documentation](https://python.langchain.com/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [HotPotQA Dataset](https://hotpotqa.github.io/)
- [Google Gemini API](https://ai.google.dev/)
