"""
AnswerMetricBundle — score generated answers against ground truth.

Black-box evaluation: the only signal is whether the generated answer
matches the expected answer.  Context is intentionally ignored.

Metrics (requires ground truth):
    answer_correctness, answer_similarity  (RAGAS, if installed)
    token_f1, rouge_l                      (always available)
    bertscore_f1                           (if bert_score is installed)
"""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


def _normalize_tokens(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace (SQuAD-style)."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()

# ---------------------------------------------------------------------------
# Optional heavy dependencies — graceful degradation
# ---------------------------------------------------------------------------
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import answer_correctness, answer_similarity
    from datasets import Dataset as HFDataset

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False

try:
    from bert_score import score as bert_score_fn

    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False


class AnswerMetricBundle:
    """
    Compute answer-correctness metrics for a batch of (question, answer, ground_truth) triples.

    All RAGAS metrics are computed in a single ``ragas.evaluate()`` call
    to minimise LLM round-trips.
    """

    RAGAS_METRICS = (
        [answer_correctness, answer_similarity]
        if _RAGAS_AVAILABLE
        else []
    )

    # ------------------------------------------------------------------ #
    # Lexical / semantic helpers                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def token_f1(pred: str, gold: str) -> float:
        """SQuAD-style token-level F1 (case-insensitive, punctuation-stripped)."""
        pred_toks = _normalize_tokens(pred)
        gold_toks = _normalize_tokens(gold)
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
        pred_toks = _normalize_tokens(pred)
        gold_toks = _normalize_tokens(gold)
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
    # Core compute                                                         #
    # ------------------------------------------------------------------ #

    def compute(
        self,
        questions: List[str],
        answers: List[str],
        ground_truths: List[str],
    ) -> List[Dict[str, float]]:
        """Score answers against ground truth.  Context is not used."""
        if not _RAGAS_AVAILABLE:
            return self._lexical_only(answers, ground_truths)

        ragas_scores: Dict[int, Dict[str, float]] = {}
        try:
            dataset = HFDataset.from_dict(
                {
                    "question": questions,
                    "answer": answers,
                    "contexts": [[] for _ in questions],
                    "ground_truth": ground_truths,
                }
            )
            result_df = ragas_evaluate(dataset, metrics=self.RAGAS_METRICS).to_pandas()
            for i, (_, row) in enumerate(result_df.iterrows()):
                ac = row.get("answer_correctness", 0.0)
                asim = row.get("answer_similarity", 0.0)
                ragas_scores[i] = {
                    "answer_correctness": 0.0 if (isinstance(ac, float) and math.isnan(ac)) else float(ac),
                    "answer_similarity": 0.0 if (isinstance(asim, float) and math.isnan(asim)) else float(asim),
                }
        except Exception:
            logger.warning("RAGAS evaluation failed; using lexical metrics only.")

        out: List[Dict[str, float]] = []
        for i in range(len(questions)):
            entry: Dict[str, float] = {
                "answer_correctness": ragas_scores.get(i, {}).get("answer_correctness", 0.0),
                "answer_similarity": ragas_scores.get(i, {}).get("answer_similarity", 0.0),
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
                "answer_correctness": 0.0,
                "answer_similarity": 0.0,
                "token_f1": self.token_f1(a, g),
                "rouge_l": self.rouge_l(a, g),
            }
            for a, g in zip(answers, ground_truths)
        ]
