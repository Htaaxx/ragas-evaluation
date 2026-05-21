# RagasFilter & FilterEvaluator - Complete Guide

## Overview

Two production-ready classes for complete RAGAS-based filter pipeline:

1. **RagasFilter** - Full training pipeline (data → RAGAS → models → inference)
2. **FilterEvaluator** - Evaluation & inference on new data

---

## RagasFilter: Complete Training Pipeline

### What It Does

```
Load CSV → Extract Contexts → Compute RAGAS → 
Train 6 Models → Select Best → Save Model
```

### Key Methods

#### 1. `load_data(csv_path)`
Load labeled data from CSV.
```python
filter = RagasFilter()
df = filter.load_data("data/labeled_asqa.csv")
# Requires columns: id, question, answer, context, supporting_facts, label
```

#### 2. `extract_contexts()`
Extract supporting facts for RAGAS.
```python
filter.extract_contexts()
# Creates filter.df["ragas_contexts"]
```

#### 3. `compute_ragas_features(ragas_evaluator)`
Compute RAGAS metrics and prepare features.
```python
from src.evaluation.ragas_evaluator import RAGASEvaluator

evaluator = RAGASEvaluator(
    metrics=["faithfulness", "answer_relevancy", "context_relevancy"],
    llm_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
)

feature_df = filter.compute_ragas_features(evaluator)
# Creates filter.feature_df with RAGAS metrics
```

#### 4. `train_models()`
Train & compare 6 classifiers.
```python
comparison_df = filter.train_models()
# Trains all 6 models, selects best by F1 → Accuracy → ROC-AUC
# Returns: DataFrame with all models' metrics
```

#### 5. `save_model()`
Save best model to disk.
```python
model_path = filter.save_model()
# Saves to: models/ragas_filter/gradient_boosting.joblib
```

#### 6. `get_feature_importance()`
Extract feature importance from best model.
```python
importance_df = filter.get_feature_importance()
# Returns: DataFrame with importance scores and percentages
```

#### 7. `train()` - Full Pipeline
Run complete pipeline from start to finish.
```python
results = filter.train(
    csv_path="data/labeled_asqa.csv",
    ragas_evaluator=evaluator,
)

# Returns dict with:
# - feature_df: Feature table
# - model_comparison: All models metrics
# - feature_importance: Importance scores
# - best_model: Best model name
# - model_path: Path to saved model
```

### Complete Example: Training

```python
from src.filtering import RagasFilter
from src.evaluation.ragas_evaluator import RAGASEvaluator

# 1. Initialize
filter_pipeline = RagasFilter(
    output_dir="results/my_filter",
    model_dir="models/my_filter",
)

# 2. Setup RAGAS
evaluator = RAGASEvaluator(
    metrics=["faithfulness", "answer_relevancy", "context_relevancy"],
    llm_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
)

# 3. Train (A-Z)
results = filter_pipeline.train(
    csv_path="data/labeled_asqa.csv",
    ragas_evaluator=evaluator,
)

# 4. Check results
print(f"Best model: {results['best_model']}")
print(f"Model saved to: {results['model_path']}")
print(f"\nModel Comparison:\n{results['model_comparison']}")
print(f"\nFeature Importance:\n{results['feature_importance']}")
```

---

## FilterEvaluator: Inference & Evaluation

### What It Does

```
Load Model → Apply to New Data → Generate Predictions → 
Save Results → Evaluate Performance
```

### Key Methods

#### 1. `load_model()`
Load trained model from disk.
```python
evaluator = FilterEvaluator(
    model_dir="models/my_filter",
    output_dir="results/my_filter/eval",
)

model = evaluator.load_model()
# Loads: models/my_filter/gradient_boosting.joblib
```

#### 2. `predict(ragas_df, data_df=None)`
Generate accept/reject predictions.
```python
# On new test data
predictions_df = evaluator.predict(
    ragas_df=test_ragas_metrics,  # RAGAS features only
    data_df=test_data,             # Optional: original data for context
)

# Returns DataFrame with columns:
# - filter_label: 0 (reject) or 1 (accept)
# - filter_confidence: Confidence score [0, 1]
# - Plus all columns from data_df if provided
```

#### 3. `save_predictions(output_name="filtered_predictions.csv")`
Save predictions to CSV.
```python
output_path = evaluator.save_predictions()
# Saves to: results/my_filter/eval/filtered_predictions.csv
```

#### 4. `evaluate_classification(y_true)`
Evaluate classification performance.
```python
metrics = evaluator.evaluate_classification(y_true=test_labels)

# Returns dict with:
# - accuracy: Classification accuracy
# - precision: Precision score
# - recall: Recall score
# - f1: F1 score
# - roc_auc: ROC-AUC score
```

### Complete Example: Evaluation

```python
from src.filtering import FilterEvaluator

# 1. Initialize
evaluator = FilterEvaluator(
    model_dir="models/my_filter",
    output_dir="results/my_filter/eval",
)

# 2. Load model
evaluator.load_model()

# 3. Generate predictions
predictions_df = evaluator.predict(
    ragas_df=test_ragas_metrics,
    data_df=test_data,
)

# 4. Save predictions
evaluator.save_predictions()

# 5. Evaluate (if labels available)
metrics = evaluator.evaluate_classification(y_true=test_labels)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

---

## Complete Workflow: Train + Evaluate

```python
from src.filtering import RagasFilter, FilterEvaluator
from src.evaluation.ragas_evaluator import RAGASEvaluator
import pandas as pd

# ============================================================
# PHASE 1: TRAIN
# ============================================================

# Setup
filter_train = RagasFilter(
    output_dir="results/ragas_filter",
    model_dir="models/ragas_filter",
)

evaluator = RAGASEvaluator(
    metrics=["faithfulness", "answer_relevancy", "context_relevancy"],
    llm_model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
)

# Train
results = filter_train.train(
    csv_path="data/labeled_asqa.csv",
    ragas_evaluator=evaluator,
)

print(f"✓ Training complete!")
print(f"  Best model: {results['best_model']}")

# ============================================================
# PHASE 2: EVALUATE
# ============================================================

# Compute RAGAS for test data
test_data = pd.read_csv("data/test.csv")
test_ragas = evaluator.evaluate(
    questions=test_data["question"].tolist(),
    answers=test_data["answer"].tolist(),
    contexts=test_data["contexts"].tolist(),
).to_pandas()

# Apply filter
filter_eval = FilterEvaluator(
    model_dir="models/ragas_filter",
    output_dir="results/ragas_filter/eval",
)

predictions = filter_eval.predict(
    ragas_df=test_ragas,
    data_df=test_data,
)

filter_eval.save_predictions()

# Evaluate
if "label" in test_data.columns:
    metrics = filter_eval.evaluate_classification(
        y_true=test_data["label"].values
    )
```

---

## Output Structure

### After Training (RagasFilter)

```
results/ragas_filter/
├── ragas_features.csv           # Feature table (labeled data)
├── model_comparison.csv         # 6 models × 5 metrics
├── feature_importance.csv       # Feature importance
├── ragas_checkpoints.csv        # Checkpoint for resuming
└── (stored on disk)

models/ragas_filter/
└── gradient_boosting.joblib     # Trained best model
```

### After Evaluation (FilterEvaluator)

```
results/ragas_filter/eval/
├── filtered_predictions.csv     # Predictions on test data
└── (stored on disk)
```

---

## Key Features

### RagasFilter
✅ Complete training pipeline (data → model)  
✅ Supports 6 classifiers (auto-comparison)  
✅ Smart model selection (by F1 → Accuracy → ROC-AUC)  
✅ RAGAS metric computation with checkpointing  
✅ Feature importance extraction  
✅ Automatic model persistence  
✅ Progress tracking throughout  

### FilterEvaluator
✅ Easy model loading  
✅ Confidence scores (not just predictions)  
✅ Classification performance metrics  
✅ Batch prediction capability  
✅ Result persistence to CSV  
✅ Full traceability (original data + predictions)  

---

## Configuration

### RagasFilter Options

```python
filter = RagasFilter(
    output_dir="./results/ragas_filter",     # Where to save results
    model_dir="./models/ragas_filter",       # Where to save models
    test_size=0.2,                           # Train-test split ratio
    random_state=42,                         # Reproducibility
)
```

### FilterEvaluator Options

```python
evaluator = FilterEvaluator(
    model_dir="./models/ragas_filter",       # Where models are saved
    output_dir="./results/ragas_filter",     # Where to save results
)
```

---

## Supported Models

### Classifiers (trained automatically)

1. **Logistic Regression** - Simple baseline
2. **Random Forest** (400 trees) - Ensemble method
3. **Gradient Boosting** ← Usually best (ranked by F1)
4. **HistGradient Boosting** - Fast alternative
5. **Extra Trees** (500 trees) - Robust ensemble
6. **XGBoost** - If installed

### Model Selection

Models ranked by:
1. F1 Score (primary)
2. Accuracy (secondary)
3. ROC-AUC (tertiary)

Best model typically: **Gradient Boosting**

---

## Expected Results (ASQA Dataset)

| Metric | Value |
|--------|-------|
| Best Model | Gradient Boosting |
| Accuracy | ~90% |
| F1 Score | ~0.90 |
| ROC-AUC | ~0.96 |
| Acceptance Rate | ~95% |
| Top Feature | Faithfulness (~89%) |

---

## Error Handling

All errors are caught and reported with helpful messages:

```python
try:
    filter_pipeline.load_data("nonexistent.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")  # Clear error message
```

---

## Integration

### Import

```python
from src.filtering import RagasFilter, FilterEvaluator
```

### With existing RAGAS evaluator

```python
from src.evaluation.ragas_evaluator import RAGASEvaluator

evaluator = RAGASEvaluator(...)
filter_pipeline = RagasFilter()
results = filter_pipeline.train(..., ragas_evaluator=evaluator)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ImportError | `pip install -r requirements.txt` |
| Out of memory | Reduce batch_size in RAGAS evaluator |
| Model not found | Check model_dir contains `.joblib` file |
| RAGAS API errors | Verify OPENAI_API_KEY environment variable |
| CSV columns missing | Check CSV has: id, question, answer, context, supporting_facts, label |

---

## Files

- **Main module**: `src/filtering/ragas_filter.py` (630 lines)
- **RagasFilter class**: Complete training pipeline
- **FilterEvaluator class**: Inference & evaluation
- **Integration**: Updated `src/filtering/__init__.py`

---

## Quick Start

```python
# 1. Train
from src.filtering import RagasFilter
from src.evaluation.ragas_evaluator import RAGASEvaluator

filter_pipeline = RagasFilter()
evaluator = RAGASEvaluator(...)
results = filter_pipeline.train("data/labeled.csv", evaluator)

# 2. Evaluate
from src.filtering import FilterEvaluator

filter_eval = FilterEvaluator()
predictions = filter_eval.predict(test_ragas, test_data)
filter_eval.save_predictions()

# 3. Done! Model saved & ready to use
```

---

**Status**: ✅ Production Ready  
**Version**: 1.0  
**Date**: May 2026
