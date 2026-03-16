# ASQA RAG Implementation Guide

Complete implementation of RAG system on ASQA long-form QA dataset with baseline comparison and comprehensive evaluation.

## 📋 Overview

This implementation provides:
1. **RAG on ASQA Dataset**: Trained retriever + generator for long-form answers
2. **Baseline Comparison**: Normal RAG vs RAG + LLM Filter
3. **Synthetic Data Generation**: ~1000 labeled samples using LLM-as-judge
4. **Comprehensive Evaluation**: Filter effectiveness + quality metrics

## 🏗️ Implementation Structure

```
ragas-evaluation/
├── notebooks/
│   ├── asqa_data_preparation.ipynb       # Download and prepare ASQA
│   ├── rag-asqa-baseline.ipynb           # Train RAG and run baselines
│   ├── synthetic_data_generation.ipynb   # Generate labeled data
│   └── evaluation_analysis.ipynb         # Comprehensive evaluation
├── src/
│   ├── data/
│   │   └── loader.py                     # ASQALoader class
│   ├── filtering/
│   │   ├── __init__.py
│   │   └── llm_filter.py                 # LLM filtering module
│   └── evaluation/
│       └── ragas_evaluator.py            # RAGAS metrics
├── data/
│   └── asqa/
│       ├── train.csv                     # Processed training data
│       ├── dev.csv                       # Processed dev data
│       └── synthetic_labeled_train.csv   # Synthetic labeled data
└── results/
    ├── asqa_normal_rag_predictions.csv   # Normal RAG results
    ├── asqa_filtered_rag_predictions.csv # Filtered RAG results
    ├── evaluation_report.csv             # Comparison metrics
    └── evaluation_summary.txt            # Summary report
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `datasets` - HuggingFace datasets
- `ragas` - RAG evaluation framework
- `rouge-score` - ROUGE metrics
- `bert-score` - BERTScore
- `google-generativeai` - Gemini API for filtering

### 2. Setup Environment

Create `.env` file with your API key:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run Notebooks in Order

#### Step 1: Data Preparation
```bash
jupyter notebook notebooks/asqa_data_preparation.ipynb
```
- Downloads ASQA dataset from HuggingFace
- Transforms to HotpotQA-compatible format
- Saves to `data/asqa/train.csv` and `data/asqa/dev.csv`

#### Step 2: RAG Training and Baselines
```bash
jupyter notebook notebooks/rag-asqa-baseline.ipynb
```
- Trains retriever on ASQA (contrastive learning)
- Trains generator (T5-large for long-form)
- Runs Normal RAG baseline
- Runs RAG + LLM Filter baseline
- Saves predictions to `results/`

#### Step 3: Synthetic Data Generation
```bash
jupyter notebook notebooks/synthetic_data_generation.ipynb
```
- Uses LLM-as-judge to label ~1000 samples
- Filters by confidence threshold
- Balances correct/incorrect samples
- Saves to `data/asqa/synthetic_labeled_train.csv`

#### Step 4: Comprehensive Evaluation
```bash
jupyter notebook notebooks/evaluation_analysis.ipynb
```
- Calculates filter effectiveness metrics
- Runs RAGAS evaluation
- Computes ROUGE-L and BERTScore
- Generates visualizations and reports
- Saves to `results/`

## 📊 Key Components

### 1. ASQA Data Loader

**File**: `src/data/loader.py`

```python
from src.data.loader import ASQALoader

loader = ASQALoader()
df_train, df_dev = loader.load_data("data/asqa/train.csv", "data/asqa/dev.csv")
corpus_texts, doc_titles = loader.build_corpus()
```

Features:
- Parses ASQA format (ambiguous questions, long answers, Wikipedia pages)
- Creates retriever and generator training examples
- Handles long-form answers (50-200 words)

### 2. LLM Filtering Module

**File**: `src/filtering/llm_filter.py`

```python
from src.filtering.llm_filter import LLMFilterPipeline

filter_pipeline = LLMFilterPipeline(
    api_key="your_api_key",
    context_threshold=6.0,
    answer_threshold=6.0
)

# Filter contexts before generation
filtered_contexts, results = filter_pipeline.filter_contexts(question, passages)

# Filter answer after generation
answer_result = filter_pipeline.filter_answer(question, answer, contexts)
```

Features:
- **Context Filter**: Scores retrieved passages (0-10), removes irrelevant ones
- **Answer Filter**: Evaluates faithfulness, relevance, completeness
- **Async Processing**: Efficient batch filtering with rate limiting

### 3. RAGAS Evaluator

**File**: `src/evaluation/ragas_evaluator.py`

```python
from src.evaluation.ragas_evaluator import compare_rag_systems

comparison_df = compare_rag_systems(
    df_normal_rag,
    df_filtered_rag,
    system1_name="Normal RAG",
    system2_name="Filtered RAG"
)
```

Metrics:
- **Faithfulness**: Answer grounded in context
- **Answer Relevancy**: Answer addresses question
- **Context Precision**: Relevant contexts ranked higher
- **Context Recall**: All relevant contexts retrieved

## 📈 Evaluation Metrics

### Filter Effectiveness

**Context Filter**:
- Total contexts retrieved
- Contexts filtered out
- Filter rate (%)
- Average contexts before/after

**Answer Filter**:
- GOOD vs BAD distribution
- Wrong answers filtered (%)
- Correct answers retained (%)
- Precision, Recall, F1

### Quality Metrics

**RAGAS Framework**:
- Faithfulness (LLM-based)
- Answer Relevancy (embedding-based)
- Context Precision
- Context Recall

**Traditional NLG**:
- ROUGE-L (lexical overlap)
- BERTScore (semantic similarity)

## 🎯 Expected Results

### Filter Effectiveness
- Context filter rate: ~20-40%
- Answer filter rate: ~30-50% flagged as BAD
- Precision/Recall trade-off based on thresholds

### Quality Improvement
- ROUGE-L: Modest improvement (1-5%)
- BERTScore: Semantic similarity maintained or improved
- Faithfulness: Significant improvement (10-20%)
- Reduced hallucinations in filtered answers

## 🔧 Configuration

### RAG System Configuration

```python
config = RAGConfig(
    encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    generator_model="google/flan-t5-large",  # Larger for long-form
    
    retriever_epochs=5,
    retriever_batch_size=16,
    retriever_lr=2e-5,
    
    generator_epochs=3,
    generator_batch_size=2,  # Smaller for longer sequences
    generator_max_output_length=512,  # Longer outputs
    
    top_k=5,
    device="cuda"
)
```

### Filter Configuration

```python
filter_pipeline = LLMFilterPipeline(
    model_name="gemini-2.5-flash",
    context_threshold=6.0,  # Adjust for precision/recall
    answer_threshold=6.0,
    max_concurrent=10  # API rate limiting
)
```

## 📝 Output Files

### Data Files
- `data/asqa/train.csv` - Processed training data (4,353 samples)
- `data/asqa/dev.csv` - Processed dev data (948 samples)
- `data/asqa/synthetic_labeled_train.csv` - Labeled data (~1000 samples)

### Model Files
- `models/asqa_retriever_trained/` - Trained retriever
- `models/asqa_generator_trained/` - Trained generator
- `rag_output_asqa/index/` - FAISS index

### Results Files
- `results/asqa_normal_rag_predictions.csv` - Normal RAG predictions
- `results/asqa_filtered_rag_predictions.csv` - Filtered RAG predictions
- `results/evaluation_report.csv` - Metrics comparison
- `results/evaluation_summary.txt` - Text summary
- `results/comprehensive_evaluation.png` - Visualization

## 🐛 Troubleshooting

### Common Issues

**1. ASQA Dataset Download Fails**
```python
# Manual download
from datasets import load_dataset
dataset = load_dataset("din0s/asqa", trust_remote_code=True)
```

**2. Out of Memory During Training**
- Reduce `batch_size` (try 1-2 for generator)
- Use smaller model: `google/flan-t5-base` instead of `large`
- Enable gradient checkpointing

**3. LLM Filter API Rate Limits**
- Reduce `max_concurrent` (try 5 instead of 10)
- Add sleep between batches
- Use caching for repeated contexts

**4. RAGAS Evaluation Fails**
- Check API keys (OpenAI for RAGAS)
- Install latest version: `pip install --upgrade ragas`
- Use alternative metrics if API unavailable

## 📚 Key Differences from HotpotQA

| Aspect | HotpotQA | ASQA |
|--------|----------|------|
| **Answer Type** | Short-form (1-5 words) | Long-form (50-200 words) |
| **Questions** | Factoid | Ambiguous factoid |
| **Generator Model** | T5-base | T5-large/XL |
| **Max Output Length** | 128 tokens | 512 tokens |
| **Training Time** | ~2-3 hours | ~6-8 hours |
| **Evaluation Focus** | Exact match | Semantic similarity |

## 🎓 Usage for Research

### For Phuong Quynh's Tasks

This implementation provides:

1. **Synthetic Labeled Data**: Use `data/asqa/synthetic_labeled_train.csv` for training evaluation models

2. **Evaluation Framework**: Adapt the comprehensive evaluation approach for WikiEval metric design

3. **Baseline Metrics**: Use filter effectiveness and quality metrics as comparison baselines

### Handoff Points

- **Synthetic data**: ~1000 labeled samples with confidence scores
- **Evaluation results**: Filter effectiveness (precision/recall) and quality metrics
- **Evaluation framework**: RAGAS integration and comprehensive analysis pipeline

## 📖 References

- **ASQA Paper**: [ASQA: Factoid Questions Meet Long-Form Answers](https://arxiv.org/abs/2204.06092)
- **RAGAS Framework**: [RAGAS Documentation](https://docs.ragas.io/)
- **HuggingFace Dataset**: [din0s/asqa](https://huggingface.co/datasets/din0s/asqa)

## 🤝 Contributing

This implementation is part of a research project. For questions or improvements:
- Check the plan file: `.cursor/plans/asqa_rag_implementation_*.plan.md`
- Review notebooks for detailed implementation
- Coordinate with team members on evaluation metrics

## ✅ Completion Checklist

- [x] Download and prepare ASQA dataset
- [x] Implement ASQALoader class
- [x] Train retriever on ASQA
- [x] Train generator for long-form answers
- [x] Implement LLM filtering module
- [x] Run Normal RAG baseline
- [x] Run RAG + Filter baseline
- [x] Generate synthetic labeled data
- [x] Integrate RAGAS evaluation
- [x] Comprehensive evaluation and analysis
- [x] Generate comparison reports
- [x] Create visualizations

## 📊 Next Steps

1. **Threshold Tuning**: Experiment with different filter thresholds
2. **Model Comparison**: Try different retriever/generator models
3. **Error Analysis**: Deep dive into specific error categories
4. **Filter Ensemble**: Combine multiple filtering strategies
5. **Human Evaluation**: Validate LLM judge with human annotations

---

**Status**: ✅ Implementation Complete

All notebooks, modules, and evaluation scripts are ready to use. Follow the Quick Start guide to reproduce results.
