"""
Traditional NLP evaluation metrics for RAG / QA systems.

Includes:
- BLEU
- ROUGE
- Exact Match (EM)
- Token-level F1
- BERTScore

Supports:
- single prediction evaluation
- dataframe evaluation
- system comparison
"""

from __future__ import annotations

import logging
import re
import string
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# =========================
# Optional imports
# =========================
try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    from bert_score import score as bertscore_score

    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


# =========================================================
# Utilities
# =========================================================

def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation/articles/extra whitespace."""
    text = text.lower()

    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())

    return text


def token_f1(prediction: str, ground_truth: str) -> float:
    """SQuAD-style token F1."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(
        normalize_text(prediction) == normalize_text(ground_truth)
    )


# =========================================================
# Evaluator
# =========================================================

class TraditionalEvaluator:
    """Traditional QA / RAG evaluator."""

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        bert_model: str = "microsoft/deberta-xlarge-mnli",
    ) -> None:

        self.available_metrics = {
            "bleu": self.compute_bleu,
            "rougeL": self.compute_rougeL,
            "f1": token_f1,
            "exact_match": exact_match,
            "bertscore": self.compute_bertscore,
        }

        if metrics is None:
            metrics = [
                "bleu",
                "rougeL",
                "f1",
                "exact_match",
            ]

        self.metrics = metrics
        self.bert_model = bert_model

    # =====================================================
    # Metric functions
    # =====================================================

    def compute_bleu(self, pred: str, gt: str) -> float:
        if not BLEU_AVAILABLE:
            raise ImportError("nltk not installed")

        smoothie = SmoothingFunction().method4

        return sentence_bleu(
            [gt.split()],
            pred.split(),
            smoothing_function=smoothie,
        )

    def compute_rougeL(self, pred: str, gt: str) -> float:
        if not ROUGE_AVAILABLE:
            raise ImportError("rouge-score not installed")

        scorer = rouge_scorer.RougeScorer(
            ["rougeL"],
            use_stemmer=True
        )

        score = scorer.score(gt, pred)

        return score["rougeL"].fmeasure

    def compute_bertscore(
        self,
        preds: List[str],
        gts: List[str]
    ) -> Dict[str, float]:

        if not BERTSCORE_AVAILABLE:
            raise ImportError("bert-score not installed")

        P, R, F1 = bertscore_score(
            preds,
            gts,
            lang="en",
            model_type=self.bert_model,
            verbose=False,
        )

        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }

    # =====================================================
    # Main evaluation
    # =====================================================

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:

        results = {}

        # ---------- BERTScore ----------
        if "bertscore" in self.metrics:
            bert_results = self.compute_bertscore(
                predictions,
                references
            )
            results.update(bert_results)

        # ---------- Other metrics ----------
        for metric in self.metrics:

            if metric == "bertscore":
                continue

            scores = []

            for pred, gt in zip(predictions, references):

                score = self.available_metrics[metric](pred, gt)

                scores.append(score)

            results[metric] = sum(scores) / len(scores)

        return results

    # =====================================================
    # Per-sample evaluation
    # =====================================================

    def evaluate_per_sample(
        self,
        predictions: List[str],
        references: List[str],
    ) -> pd.DataFrame:

        rows = []

        # Precompute bertscore if needed
        bert_f1 = None

        if "bertscore" in self.metrics:

            _, _, F1 = bertscore_score(
                predictions,
                references,
                lang="en",
                model_type=self.bert_model,
                verbose=False,
            )

            bert_f1 = F1.tolist()

        for i, (pred, gt) in enumerate(zip(predictions, references)):

            row = {
                "prediction": pred,
                "reference": gt,
            }

            for metric in self.metrics:

                if metric == "bertscore":
                    row["bertscore_f1"] = bert_f1[i]

                else:
                    row[metric] = self.available_metrics[metric](
                        pred,
                        gt
                    )

            rows.append(row)

        return pd.DataFrame(rows)

    # =====================================================
    # DataFrame helper
    # =====================================================

    def evaluate_from_dataframe(
        self,
        df: pd.DataFrame,
        prediction_col: str = "predicted_answer",
        reference_col: str = "gold_answer",
    ) -> Dict[str, float]:

        predictions = df[prediction_col].fillna("").tolist()
        references = df[reference_col].fillna("").tolist()

        return self.evaluate(predictions, references)

    # =====================================================
    # Compare systems
    # =====================================================

    def compare_systems(
        self,
        system1_df: pd.DataFrame,
        system2_df: pd.DataFrame,
        prediction_col: str = "predicted_answer",
        reference_col: str = "gold_answer",
        system1_name: str = "System 1",
        system2_name: str = "System 2",
    ) -> pd.DataFrame:

        res1 = self.evaluate_from_dataframe(
            system1_df,
            prediction_col,
            reference_col,
        )

        res2 = self.evaluate_from_dataframe(
            system2_df,
            prediction_col,
            reference_col,
        )

        rows = []

        for metric in res1:

            delta = res2[metric] - res1[metric]

            rows.append({
                "Metric": metric,
                system1_name: round(res1[metric], 4),
                system2_name: round(res2[metric], 4),
                "Delta": round(delta, 4),
            })

        return pd.DataFrame(rows)


# =========================================================
# Convenience functions
# =========================================================

def evaluate_traditional_metrics(
    df: pd.DataFrame,
    prediction_col: str = "predicted_answer",
    reference_col: str = "gold_answer",
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:

    evaluator = TraditionalEvaluator(metrics=metrics)

    return evaluator.evaluate_from_dataframe(
        df,
        prediction_col=prediction_col,
        reference_col=reference_col,
    )


def compare_traditional_systems(
    system1_df: pd.DataFrame,
    system2_df: pd.DataFrame,
    prediction_col: str = "predicted_answer",
    reference_col: str = "gold_answer",
    metrics: Optional[List[str]] = None,
    system1_name: str = "System 1",
    system2_name: str = "System 2",
) -> pd.DataFrame:

    evaluator = TraditionalEvaluator(metrics=metrics)

    return evaluator.compare_systems(
        system1_df,
        system2_df,
        prediction_col=prediction_col,
        reference_col=reference_col,
        system1_name=system1_name,
        system2_name=system2_name,
    )