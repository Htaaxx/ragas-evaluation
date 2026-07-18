"""Sanity check: verify lexical metrics separate correct vs hallucinated answers."""

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.filtering.data_split import load_and_split
from src.filtering.metrics import AnswerMetricBundle

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    train_df, _, _ = load_and_split("data/asqa/labeled_asqa.csv")

    base_ids = train_df["id"].str.replace(r"b$", "", regex=True).unique()[:50]

    correct_f1, correct_rl = [], []
    hallu_f1, hallu_rl = [], []
    bundle = AnswerMetricBundle()

    for bid in base_ids:
        rows = train_df[train_df["id"].str.replace(r"b$", "", regex=True) == bid]
        correct_row = rows[rows["label"] == 1].iloc[0]
        hallu_row = rows[rows["label"] == 0].iloc[0]

        ground_truth = correct_row["answer"]

        correct_f1.append(bundle.token_f1(correct_row["answer"], ground_truth))
        correct_rl.append(bundle.rouge_l(correct_row["answer"], ground_truth))

        hallu_f1.append(bundle.token_f1(hallu_row["answer"], ground_truth))
        hallu_rl.append(bundle.rouge_l(hallu_row["answer"], ground_truth))

    results = {
        "correct_token_f1": {"mean": float(np.mean(correct_f1)), "std": float(np.std(correct_f1))},
        "correct_rouge_l": {"mean": float(np.mean(correct_rl)), "std": float(np.std(correct_rl))},
        "hallucinated_token_f1": {"mean": float(np.mean(hallu_f1)), "std": float(np.std(hallu_f1))},
        "hallucinated_rouge_l": {"mean": float(np.mean(hallu_rl)), "std": float(np.std(hallu_rl))},
        "gap_token_f1": float(np.mean(correct_f1) - np.mean(hallu_f1)),
        "gap_rouge_l": float(np.mean(correct_rl) - np.mean(hallu_rl)),
        "n_samples": 50,
    }

    print("\n=== METRIC SEPARATION SANITY CHECK (50 samples) ===")
    print(f"Correct answers   - token_f1: {results['correct_token_f1']['mean']:.4f} +/- {results['correct_token_f1']['std']:.4f}")
    print(f"Hallucinated ans  - token_f1: {results['hallucinated_token_f1']['mean']:.4f} +/- {results['hallucinated_token_f1']['std']:.4f}")
    print(f"Gap (token_f1): {results['gap_token_f1']:.4f}")
    print()
    print(f"Correct answers   - rouge_l:  {results['correct_rouge_l']['mean']:.4f} +/- {results['correct_rouge_l']['std']:.4f}")
    print(f"Hallucinated ans  - rouge_l:  {results['hallucinated_rouge_l']['mean']:.4f} +/- {results['hallucinated_rouge_l']['std']:.4f}")
    print(f"Gap (rouge_l):  {results['gap_rouge_l']:.4f}")

    out_path = Path("results/metric_separation_sanity_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
