"""Ablation studies: data size and max sequence length experiments."""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.filtering.data_split import load_and_split
from src.filtering.filter_evaluator import FilterEvaluator
from src.filtering.learned_filter import AnswerQualityClassifier, train_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/ablations")
MODELS_DIR = Path("models/ablations")
SEED = 42


def find_best_threshold(clf: AnswerQualityClassifier, val_df: pd.DataFrame) -> float:
    """Sweep thresholds on val set and return the one that maximises F1."""
    evaluator = FilterEvaluator()
    decisions = clf.predict_batch(val_df["question"].tolist(), val_df["answer"].tolist())
    confidences = [d.confidence for d in decisions]
    labels = val_df["label"].tolist()

    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.1, 0.95, 0.05):
        preds = [c >= t for c in confidences]
        r = evaluator.evaluate(preds, labels)
        if r.f1 > best_f1:
            best_f1 = r.f1
            best_t = round(float(t), 2)
    return best_t


def evaluate_model(clf: AnswerQualityClassifier, test_df: pd.DataFrame, threshold: float) -> dict:
    """Evaluate a model on the test set at a given threshold."""
    evaluator = FilterEvaluator()
    decisions = clf.predict_batch(test_df["question"].tolist(), test_df["answer"].tolist())
    preds = [d.confidence >= threshold for d in decisions]
    result = evaluator.evaluate(preds, test_df["label"].tolist())
    return result.to_dict()


def ablation_data_size(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Train on 25%, 50%, 75%, 100% of the training data and compare."""
    logger.info("=== ABLATION: Training Data Size ===")
    fractions = [0.25, 0.50, 0.75, 1.0]
    rows = []

    for frac in fractions:
        tag = f"data_{int(frac * 100)}pct"
        n_samples = int(len(train_df) * frac)
        subset = train_df.sample(n=n_samples, random_state=SEED).reset_index(drop=True)
        out_dir = MODELS_DIR / tag

        logger.info("Training on %d/%d samples (%.0f%%)...", n_samples, len(train_df), frac * 100)
        train_classifier(
            subset, val_df,
            output_dir=str(out_dir),
        )

        clf = AnswerQualityClassifier(str(out_dir))
        best_t = find_best_threshold(clf, val_df)
        metrics = evaluate_model(clf, test_df, best_t)
        row = {"fraction": frac, "n_train": n_samples, "threshold": best_t, **metrics}
        rows.append(row)
        logger.info("  frac=%.0f%% -> F1=%.4f Acc=%.4f (t=%.2f)", frac * 100, metrics["f1"], metrics["accuracy"], best_t)

    out_path = RESULTS_DIR / "ablation_data_size.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info("Data size ablation saved to %s", out_path)


def ablation_max_length(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Train with max_length 256, 384, 512 and compare."""
    logger.info("=== ABLATION: Max Sequence Length ===")
    lengths = [256, 384, 512]
    rows = []

    for ml in lengths:
        tag = f"maxlen_{ml}"
        out_dir = MODELS_DIR / tag

        logger.info("Training with max_length=%d...", ml)
        train_classifier(
            train_df, val_df,
            output_dir=str(out_dir),
            config_overrides={"max_length": ml},
        )

        clf = AnswerQualityClassifier(str(out_dir))
        clf.max_length = ml
        best_t = find_best_threshold(clf, val_df)
        metrics = evaluate_model(clf, test_df, best_t)
        row = {"max_length": ml, "threshold": best_t, **metrics}
        rows.append(row)
        logger.info("  max_length=%d -> F1=%.4f Acc=%.4f (t=%.2f)", ml, metrics["f1"], metrics["accuracy"], best_t)

    out_path = RESULTS_DIR / "ablation_max_length.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info("Max length ablation saved to %s", out_path)


def main() -> None:
    train_df, val_df, test_df = load_and_split("data/asqa/labeled_asqa.csv")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ablation_data_size(train_df, val_df, test_df)
    ablation_max_length(train_df, val_df, test_df)

    logger.info("All ablations complete.")


if __name__ == "__main__":
    main()
