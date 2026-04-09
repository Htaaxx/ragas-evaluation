"""
HybridMetricBundle — compute up to 9 quality metrics for RAG evaluation.

Proxy bundle (reference-free, inference-safe, 3 metrics):
    faithfulness, answer_relevancy, context_relevancy

Full bundle (requires ground truth, calibration only, up to 9 metrics):
    + context_precision, context_recall, answer_correctness,
      answer_similarity, token_f1, rouge_l
    + bertscore_f1 if bert_score package is installed
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies — graceful degradation
# ---------------------------------------------------------------------------
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        answer_similarity,
        context_precision,
        context_recall,
        context_relevancy,
        faithfulness,
    )
    from datasets import Dataset as HFDataset

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False

try:
    from bert_score import score as bert_score_fn

    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False


class HybridMetricBundle:
    """
    Compute up to 9 quality metrics for a batch of (q, a, C, [gt]) triples.

    All RAGAS metrics are computed in a single ``ragas.evaluate()`` call per
    mode to minimise LLM round-trips.
    """

    PROXY_RAGAS = (
        [faithfulness, answer_relevancy, context_relevancy]
        if _RAGAS_AVAILABLE
        else []
    )
    FULL_RAGAS = (
        [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
            answer_similarity,
            context_relevancy,
        ]
        if _RAGAS_AVAILABLE
        else []
    )

    # ------------------------------------------------------------------ #
    # Static lexical / semantic helpers                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def token_f1(pred: str, gold: str) -> float:
        """SQuAD-style token-level F1 (case-insensitive, whitespace split)."""
        pred_toks = pred.lower().split()
        gold_toks = gold.lower().split()
        if not pred_toks or not gold_toks:
            return 0.0
        common = set(pred_toks) & set(gold_toks)
        if not common:
            return 0.0
        prec = len(common) / len(pred_toks)
        rec = len(common) / len(gold_toks)
        return 2.0 * prec * rec / (prec + rec)

    @staticmethod
    def rouge_l(pred: str, gold: str) -> float:
        """ROUGE-L F1 via dynamic programming on LCS."""
        pred_toks = pred.lower().split()
        gold_toks = gold.lower().split()
        if not pred_toks or not gold_toks:
            return 0.0
        m, n = len(gold_toks), len(pred_toks)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if gold_toks[i - 1] == pred_toks[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        prec = lcs / n
        rec = lcs / m
        if prec + rec == 0:
            return 0.0
        return 2.0 * prec * rec / (prec + rec)

    # ------------------------------------------------------------------ #
    # Compute methods                                                       #
    # ------------------------------------------------------------------ #

    def compute_proxy(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
    ) -> List[Dict[str, float]]:
        """Reference-free metric bundle (3 metrics). Safe at inference time."""
        if not _RAGAS_AVAILABLE:
            return [
                {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_relevancy": 0.0,
                }
                for _ in questions
            ]

        dataset = HFDataset.from_dict(
            {"question": questions, "answer": answers, "contexts": contexts}
        )
        result_df = ragas_evaluate(dataset, metrics=self.PROXY_RAGAS).to_pandas()

        out: List[Dict[str, float]] = []
        for _, row in result_df.iterrows():
            out.append(
                {
                    "faithfulness": float(row.get("faithfulness", 0.0)),
                    "answer_relevancy": float(row.get("answer_relevancy", 0.0)),
                    "context_relevancy": float(row.get("context_relevancy", 0.0)),
                }
            )
        return out

    def compute_full(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> List[Dict[str, float]]:
        """Full metric bundle (up to 9 metrics). Requires ground truth."""
        if not _RAGAS_AVAILABLE:
            return self._lexical_only(answers, ground_truths)

        dataset = HFDataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result_df = ragas_evaluate(dataset, metrics=self.FULL_RAGAS).to_pandas()

        out: List[Dict[str, float]] = []
        for i, (_, row) in enumerate(result_df.iterrows()):
            entry: Dict[str, float] = {
                "faithfulness": float(row.get("faithfulness", 0.0)),
                "answer_relevancy": float(row.get("answer_relevancy", 0.0)),
                "context_precision": float(row.get("context_precision", 0.0)),
                "context_recall": float(row.get("context_recall", 0.0)),
                "answer_correctness": float(row.get("answer_correctness", 0.0)),
                "answer_similarity": float(row.get("answer_similarity", 0.0)),
                "context_relevancy": float(row.get("context_relevancy", 0.0)),
                "token_f1": self.token_f1(answers[i], ground_truths[i]),
                "rouge_l": self.rouge_l(answers[i], ground_truths[i]),
            }
            out.append(entry)

        if _BERTSCORE_AVAILABLE:
            try:
                _, _, f1s = bert_score_fn(
                    answers, ground_truths, lang="en", verbose=False
                )
                for i, entry in enumerate(out):
                    entry["bertscore_f1"] = float(f1s[i])
            except Exception:
                logger.warning("BERTScore computation failed; skipping.")

        return out

    def _lexical_only(
        self, answers: List[str], ground_truths: List[str]
    ) -> List[Dict[str, float]]:
        """Fallback when RAGAS is not installed — lexical metrics only."""
        return [
            {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0,
                "answer_similarity": 0.0,
                "context_relevancy": 0.0,
                "token_f1": self.token_f1(a, g),
                "rouge_l": self.rouge_l(a, g),
            }
            for a, g in zip(answers, ground_truths)
        ]
