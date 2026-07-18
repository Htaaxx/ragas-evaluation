"""Apply the trained filter to real RAG predictions."""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.filtering.learned_filter import AnswerQualityClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    clf = AnswerQualityClassifier("models/answer_filter")
    df = pd.read_csv("results/asqa_normal_rag_predictions.csv")
    logger.info("Loaded %d RAG predictions", len(df))

    decisions = clf.predict_batch(
        df["question"].tolist(),
        df["predicted_answer"].tolist(),
    )

    df["filter_accept"] = [d.accept for d in decisions]
    df["filter_confidence"] = [round(d.confidence, 4) for d in decisions]

    n_accept = sum(d.accept for d in decisions)
    n_reject = len(decisions) - n_accept
    logger.info("Accepted: %d (%.1f%%), Rejected: %d (%.1f%%)",
                n_accept, 100 * n_accept / len(decisions),
                n_reject, 100 * n_reject / len(decisions))

    df.to_csv("results/rag_predictions_filtered.csv", index=False)
    logger.info("Saved to results/rag_predictions_filtered.csv")


if __name__ == "__main__":
    main()