# Basic RAG System for HotPotQA

A simple and effective **Retrieval-Augmented Generation (RAG)** system implemented for the HotPotQA dataset. This project demonstrates how to build a question-answering system that retrieves relevant context and generates accurate answers.

## 🎯 What is This?

This is a **basic RAG implementation** that:
- Loads the HotPotQA multi-hop question answering dataset
- Creates a searchable vector database (FAISS) from documents
- Retrieves relevant context for any question
- Generates answers using a language model (Flan-T5)
- Provides evaluation capabilities

**Perfect for:** Learning RAG concepts, building QA systems, experimenting with retrieval methods.

## 🚀 Quick Start

### Option 1: Interactive Mode (Easiest! ⭐)

Simply run the main interface:
```bash
# Activate virtual environment first
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac

# Run interactive interface
python main.py
```

You'll see a friendly menu to guide you through:
1. Building the vector store
2. Asking questions
3. Running evaluations
4. Viewing configuration

**Perfect for beginners!** The interactive mode will prompt you for all settings.

### Option 2: Direct Mode Selection

Run a specific mode directly:
```bash
python main.py --mode build      # Build vector store
python main.py --mode query      # Ask questions
python main.py --mode evaluate   # Run evaluation
python main.py --mode ragas      # RAGAS evaluation
python main.py --config          # Show settings
```

### Option 3: Advanced CLI (For Power Users)

Use the advanced CLI for full control:
```bash
# Build with custom settings
python src/run_rag.py --mode build --max-samples 1000 --chunk-size 500

# Query with specific LLM
python src/run_rag.py --mode query --use-gemini --top-k 5

# RAGAS evaluation
python src/run_rag.py --mode ragas --num-questions 10 --use-gemini
```

### Setup Requirements

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Set API Keys:**

Get your **Google Gemini API key** (recommended): https://aistudio.google.com/apikey

```bash
# Windows (PowerShell):
$env:GOOGLE_API_KEY="your_key_here"

# Linux/Mac:
export GOOGLE_API_KEY="your_key_here"

# Or create .env file:
echo "GOOGLE_API_KEY=your_key_here" > .env
```

**3. Ensure Dataset is Present:**
```
data/hotpot_dev_distractor_v1.json
```

### Example Workflow

**First time:**
```bash
python main.py
# Select: 1. Build Vector Store
# Follow the prompts (defaults work great!)
```

**Daily usage:**
```bash
python main.py
# Select: 2. Interactive Query
# Ask your questions!
```

**📖 See [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed instructions and examples.**

### Additional Commands

**View configuration:**
```bash
python main.py --config
```

**Run batch evaluation:**
```bash
python main.py --mode evaluate
# Or use advanced CLI:
python src/run_rag.py --mode evaluate --num-questions 10
```

**Run RAGAS evaluation:**
```bash
python main.py --mode ragas
# Or use advanced CLI:
python src/run_rag.py --mode ragas --num-questions 5 --use-gemini
```

**RAGAS Metrics:**
- **Faithfulness**: Is the answer grounded in retrieved context?
- **Answer Relevancy**: Is the answer relevant to the question?
- **Context Precision**: How relevant are retrieved documents?
- **Context Recall**: Was all relevant information retrieved?
- **Answer Correctness**: Overall quality vs ground truth
- **Answer Similarity**: Semantic similarity to ground truth

📖 **See [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed instructions and [RAGAS_GUIDE.md](RAGAS_GUIDE.md) for RAGAS details**

## ✨ Features

- ✅ **Modular Architecture**: Clean, maintainable code structure
- ✅ **Simple API**: Easy-to-use Python interface
- ✅ **HotPotQA Dataset**: Multi-hop question answering with 7,405 questions
- ✅ **FAISS Vector Store**: Fast similarity search
- ✅ **Multiple LLM Providers**: HuggingFace Inference API and Google Gemini
- ✅ **Four Modes**: Build, Query, Evaluate, and RAGAS Evaluation
- ✅ **RAGAS Integration**: Comprehensive RAG-specific evaluation metrics
- ✅ **Configurable**: Adjust models, chunk sizes, retrieval parameters
- ✅ **Well Documented**: Comprehensive docs and architecture guides

## 🏗️ Architecture

```
User Question
     ↓
[Embedding Model] → Convert question to vector
     ↓
[FAISS Similarity Search] → Find top-k relevant document chunks
     ↓
[Retrieved Context] → Pass context to LLM
     ↓
[Language Model (Flan-T5)] → Generate answer based on context
     ↓
Final Answer
```

## 📦 Project Structure

```
ragas-evaluation/
├── main.py                      # ⭐ Main entry point (interactive interface)
├── README.md                    # This file
├── USAGE_GUIDE.md              # Detailed usage instructions
├── requirements.txt             # Python dependencies
├── example_ragas_evaluation.py  # RAGAS usage examples
├── data/
│   └── hotpot_dev_distractor_v1.json  # HotPotQA dataset
├── vectorstore/                 # Generated vector store (after build)
│   ├── index.faiss             # FAISS index
│   ├── index.pkl               # Metadata
│   └── questions.json          # Questions from dataset
└── src/
    ├── README.md               # Module documentation
    ├── config.py               # Configuration settings
    ├── rag_system.py           # Main RAG orchestrator
    ├── run_rag.py              # Advanced CLI interface
    ├── models/                 # LLM wrappers
    │   ├── huggingface_llm.py # HuggingFace Inference API
    │   └── gemini_llm.py      # Google Gemini API
    ├── data/                   # Data loading
    │   └── loader.py          # HotPotQA loader
    ├── vectorstore/           # Vector store management
    │   └── manager.py         # FAISS manager
    ├── evaluation/            # Evaluation metrics
    │   └── ragas_evaluator.py # RAGAS framework
    ├── utils/                 # Utilities
    │   └── file_utils.py     # File I/O helpers
    └── basic_rag.py          # DEPRECATED (kept for compatibility)
```

**📖 New Features:**
- **`main.py`** - Interactive interface with friendly menus (perfect for beginners!)
- **Modular Architecture** - Clean, maintainable code structure
- **Multiple Interfaces** - Choose interactive mode, direct commands, or advanced CLI

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed instructions and [src/README.md](src/README.md) for architecture details.

## 🎮 Command Line Options

### Build Mode
```bash
python src/run_rag.py --mode build \
  --local-file data/hotpot_dev_distractor_v1.json \  # Dataset file (default)
  --max-samples 1000 \                                # Limit samples (optional)
  --chunk-size 500 \                                  # Text chunk size
  --chunk-overlap 50 \                                # Chunk overlap
  --embedding-model sentence-transformers/all-MiniLM-L6-v2  # Embedding model
```

### Query Mode
```bash
python src/run_rag.py --mode query \
  --vectorstore-path ./vectorstore \                  # Path to vectorstore
  --llm-model google/flan-t5-base \                   # LLM model
  --top-k 3                                           # Number of docs to retrieve
```

### Evaluate Mode
```bash
python src/run_rag.py --mode evaluate \
  --num-questions 100 \                               # Number of questions (optional)
  --output-file results.json                          # Output file
```

### RAGAS Evaluation Mode
```bash
python src/run_rag.py --mode ragas \
  --num-questions 50 \                                # Number of questions (optional)
  --ragas-metrics faithfulness answer_relevancy \     # Specific metrics (optional)
  --output-file ragas_results.json                    # Output file
```

**Available RAGAS metrics:** `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`, `answer_correctness`, `answer_similarity`

## 📊 Example Results

### Basic Evaluation
```
Question: What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?
Ground Truth: 6.213 km long
RAG Answer: 6.213 km long
Retrieved: 3 documents

Question: Were Scott Derrickson and Ed Wood of the same nationality?
Ground Truth: yes
RAG Answer: yes
Retrieved: 3 documents
```

### RAGAS Evaluation
```
Overall RAGAS Scores:
  faithfulness: 0.8542
  answer_relevancy: 0.7891
  context_precision: 0.7234
  context_recall: 0.8012
  answer_correctness: 0.7654
  answer_similarity: 0.8123

Total samples: 50
```

## 🛠️ Requirements

- Python 3.8+
- 4GB+ RAM (8GB+ recommended for full dataset)
- HuggingFace API token (free) - **That's all you need!**
- HotPotQA dataset in `data/` folder
- ❌ **No OpenAI API key required** for RAGAS evaluation

See [requirements.txt](requirements.txt) for full dependency list.

## 📖 How It Works

The RAG system follows these steps:

1. **Load Data**: Read HotPotQA dataset from local JSON file
2. **Chunk Documents**: Split long documents into 500-character chunks with 50-character overlap
3. **Create Embeddings**: Convert text chunks to 384-dimensional vectors using sentence-transformers
4. **Build Index**: Create FAISS vector store for fast similarity search
5. **Retrieve**: Given a question, find top-k most similar document chunks
6. **Generate**: Pass retrieved context to Flan-T5 LLM to generate answer

## 🎯 Use Cases

- **Question Answering**: Build QA systems on custom data
- **Learning**: Understand RAG concepts hands-on
- **Experimentation**: Test different models and parameters
- **Baseline**: Starting point for more advanced RAG systems

## 🔍 Dataset: HotPotQA

HotPotQA is a question answering dataset featuring:
- **Multi-hop reasoning**: Questions requiring information from multiple documents
- **Wikipedia context**: Real-world documents
- **Diverse questions**: Various types and difficulty levels
- **7,405 questions** in the dev_distractor split

Example:
```
Question: "What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?"
Requires: Finding where the event was held → Finding the track length
Answer: "6.213 km long"
```

## 🚧 Limitations

- **Context window**: Limited by LLM max tokens (~512 tokens)
- **Single retrieval step**: No iterative refinement
- **No re-ranking**: Uses top-k documents directly
- **Static knowledge**: Requires rebuilding vectorstore for updates
- **Simple chunking**: Fixed-size chunks may split important information

## 🔮 Recent Updates & Future Improvements

### ✅ Completed
- [x] ~~Add RAGAS evaluation framework~~ ✅ **DONE!**
- [x] ~~Refactor to modular architecture~~ ✅ **DONE!**
- [x] ~~Add Google Gemini support~~ ✅ **DONE!**
- [x] ~~Comprehensive documentation~~ ✅ **DONE!**

### 🚀 Planned
- [ ] Add re-ranking for better retrieval accuracy
- [ ] Implement hybrid search (semantic + keyword/BM25)
- [ ] Add support for more LLM providers (Anthropic Claude, etc.)
- [ ] Implement iterative retrieval for complex questions
- [ ] Add answer verification/confidence scoring
- [ ] Support streaming responses
- [ ] Add web UI (FastAPI/Gradio)
- [ ] Add caching layer for LLM responses

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

This is a basic implementation for learning purposes. Feel free to fork and enhance!

## 📧 Support

For questions or issues, please check:
1. Is your HuggingFace API token set correctly?
2. Is the dataset file present in `data/` folder?
3. Did you build the vectorstore before querying?
4. Are you activating the virtual environment?

---

**Built with:** LangChain, FAISS, HuggingFace Transformers, Sentence Transformers, RAGAS

**New to RAGAS?** Check out [RAGAS_GUIDE.md](RAGAS_GUIDE.md) for a comprehensive guide on evaluating your RAG system!
