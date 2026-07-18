"""Threshold tuning on val set, final evaluation on test set, comparison table."""

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.filtering.data_split import load_and_split
from src.filtering.filter_evaluator import FilterEvaluator
from src.filtering.learned_filter import AnswerQualityClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    _, val_df, test_df = load_and_split("data/asqa/labeled_asqa.csv")
    clf = AnswerQualityClassifier("models/answer_filter")
    evaluator = FilterEvaluator()

    # --- Threshold tuning on validation set ---
    logger.info("Running predictions on validation set (%d samples)...", len(val_df))
    val_decisions = clf.predict_batch(val_df["question"].tolist(), val_df["answer"].tolist())
    val_confidences = [d.confidence for d in val_decisions]
    val_labels = val_df["label"].tolist()

    best_f1, best_threshold = 0.0, 0.5
    threshold_results = []
    for t in np.arange(0.1, 0.95, 0.05):
        preds = [c >= t for c in val_confidences]
        result = evaluator.evaluate(preds, val_labels)
        threshold_results.append({"threshold": round(float(t), 2), **result.to_dict()})
        if result.f1 > best_f1:
            best_f1 = result.f1
            best_threshold = round(float(t), 2)

    logger.info("Best threshold: %.2f (val F1=%.4f)", best_threshold, best_f1)

    Path("results").mkdir(exist_ok=True)
    with open("results/threshold_sweep.json", "w") as f:
        json.dump(threshold_results, f, indent=2)

    # --- Final evaluation on held-out test set ---
    logger.info("Running predictions on test set (%d samples)...", len(test_df))
    test_decisions = clf.predict_batch(test_df["question"].tolist(), test_df["answer"].tolist())
    test_preds = [d.confidence >= best_threshold for d in test_decisions]
    test_labels = test_df["label"].tolist()

    learned_result = evaluator.evaluate(test_preds, test_labels)
    learned_result.save("results/learned_filter_test_results.json")

    baseline_result = evaluator.compute_no_filter_baseline(test_labels)

    # --- Comparison table ---
    comparison = evaluator.compare(
        {"No Filter": baseline_result, "Learned Filter": learned_result},
        save_path="results/filter_comparison.json",
    )
    print("\n=== COMPARISON TABLE ===")
    for row in comparison:
        print(f"\n{row['strategy']}:")
        for k, v in row.items():
            if k != "strategy":
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- Per-sample predictions for error analysis ---
    rows = []
    for i, (_, sample) in enumerate(test_df.iterrows()):
        rows.append({
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"][:200],
            "label": int(sample["label"]),
            "predicted": bool(test_preds[i]),
            "confidence": round(test_decisions[i].confidence, 4),
        })

    import pandas as pd
    pd.DataFrame(rows).to_csv("results/test_predictions.csv", index=False)
    logger.info("Per-sample predictions saved to results/test_predictions.csv")
    logger.info("Done. Best threshold=%.2f", best_threshold)


if __name__ == "__main__":
    main()