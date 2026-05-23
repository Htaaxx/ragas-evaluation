"""
Traditional NLP evaluation metrics for RAG / QA systems.

Dataset-agnostic evaluator for answer-quality evaluation.

Supported metrics:
- BLEU
- ROUGE-1 / ROUGE-2 / ROUGE-L
- String Exact Match (str_em / exact_match)
- Token-level F1
- BERTScore
- MAUVE
- Citation precision / citation recall

Notes
-----
Citation precision/recall here is a lightweight, deterministic evaluator.
It supports two cases:

1. If `citation_precision` and `citation_recall` columns already exist in the
   input dataframe, those values are used directly.

2. Otherwise, it estimates citation/support from answer-context overlap:
   - citation_precision: fraction of answer tokens supported by context
   - citation_recall: fraction of reference/gold-context-supported tokens
     covered by the answer

This is intentionally model-free. If you later implement an LLM-based citation
checker, keep the same public methods and replace the internal logic.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
import string
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =========================
# Optional imports
# =========================

try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    BLEU_AVAILABLE = True
except ImportError:  # pragma: no cover
    BLEU_AVAILABLE = False

try:
    from bert_score import score as bertscore_score
    from transformers import AutoTokenizer

    BERTSCORE_AVAILABLE = True
except ImportError:  # pragma: no cover
    BERTSCORE_AVAILABLE = False

try:
    import mauve

    MAUVE_AVAILABLE = True
except ImportError:  # pragma: no cover
    MAUVE_AVAILABLE = False


# =========================================================
# Utilities
# =========================================================

def normalize_text(text: Any) -> str:
    """Lowercase, remove English articles, punctuation, and extra whitespace."""
    text = "" if text is None else str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    return " ".join(text.split())


def safe_parse_context(context: Any) -> List[str]:
    """Parse context to list[str].

    Supports:
    - list[str]
    - list[dict] with text/content/passage/context/body keys
    - JSON/Python-list strings
    - plain strings
    """
    if context is None:
        return []
    if isinstance(context, float) and math.isnan(context):
        return []

    if isinstance(context, list):
        return [_context_item_to_text(x) for x in context if _context_item_to_text(x)]

    if isinstance(context, dict):
        text = _context_item_to_text(context)
        return [text] if text else []

    if isinstance(context, str):
        s = context.strip()
        if not s:
            return []

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(s)
                if isinstance(parsed, list):
                    return [_context_item_to_text(x) for x in parsed if _context_item_to_text(x)]
                if isinstance(parsed, dict):
                    text = _context_item_to_text(parsed)
                    return [text] if text else []
            except Exception:
                pass

        return [s]

    return [str(context)]


def _context_item_to_text(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("text", "content", "passage", "context", "body", "document"):
            if key in item and item[key] is not None:
                return str(item[key]).strip()
        return json.dumps(item, ensure_ascii=False)
    return str(item).strip()


def truncate_text(text: Any, tokenizer: Any, max_tokens: int = 256) -> str:
    text = "" if text is None else str(text)
    ids = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_tokens,
    )
    return tokenizer.decode(ids, skip_special_tokens=True)


def token_f1(prediction: Any, ground_truth: Any) -> float:
    """SQuAD-style token F1."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return float(2 * precision * recall / (precision + recall))


def exact_match(prediction: Any, ground_truth: Any) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def str_em(prediction: Any, ground_truth: Any) -> float:
    """Alias for normalized exact match."""
    return exact_match(prediction, ground_truth)


def _mean(values: Sequence[float]) -> float:
    values = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(values)) if values else np.nan


# =========================================================
# Evaluator
# =========================================================

class TraditionalEvaluator:
    """Traditional QA / RAG evaluator.

    Parameters
    ----------
    metrics:
        List of metric names. Supported:
        bleu, rouge1, rouge2, rougeL, rouge, f1, exact_match, str_em,
        bertscore, mauve, citation_precision, citation_recall, citation
    bert_model:
        BERTScore model name.
    bert_max_tokens:
        Max tokens per text before BERTScore truncation.
    mauve_max_texts:
        MAUVE can be slow. This caps the number of examples used.
    citation_min_token_len:
        Ignore tiny stopword-like tokens in citation overlap.
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        bert_model: str = "microsoft/deberta-xlarge-mnli",
        bert_max_tokens: int = 256,
        mauve_max_texts: int = 500,
        citation_min_token_len: int = 3,
    ) -> None:
        if metrics is None:
            metrics = ["bleu", "rougeL", "f1", "exact_match"]

        self.metrics = metrics
        self.bert_model = bert_model
        self.bert_max_tokens = bert_max_tokens
        self.mauve_max_texts = mauve_max_texts
        self.citation_min_token_len = citation_min_token_len

        self.available_metrics = {
            "bleu": self.compute_bleu,
            "rouge1": self.compute_rouge1,
            "rouge2": self.compute_rouge2,
            "rougeL": self.compute_rougeL,
            "f1": token_f1,
            "exact_match": exact_match,
            "str_em": str_em,
        }

    # =====================================================
    # Single-pair metric functions
    # =====================================================

    def compute_bleu(self, pred: Any, gt: Any) -> float:
        if not BLEU_AVAILABLE:
            raise ImportError("nltk is not installed. Install with: pip install nltk")

        pred = "" if pred is None else str(pred)
        gt = "" if gt is None else str(gt)
        smoothie = SmoothingFunction().method4

        return float(
            sentence_bleu(
                [gt.split()],
                pred.split(),
                smoothing_function=smoothie,
            )
        )

    def _compute_rouge(self, pred: Any, gt: Any, metric: str) -> float:
        if not ROUGE_AVAILABLE:
            raise ImportError("rouge-score is not installed. Install with: pip install rouge-score")

        scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
        score = scorer.score("" if gt is None else str(gt), "" if pred is None else str(pred))
        return float(score[metric].fmeasure)

    def compute_rouge1(self, pred: Any, gt: Any) -> float:
        return self._compute_rouge(pred, gt, "rouge1")

    def compute_rouge2(self, pred: Any, gt: Any) -> float:
        return self._compute_rouge(pred, gt, "rouge2")

    def compute_rougeL(self, pred: Any, gt: Any) -> float:
        return self._compute_rouge(pred, gt, "rougeL")

    # =====================================================
    # Batch metric functions
    # =====================================================

    def compute_bertscore(
        self,
        preds: List[str],
        gts: List[str],
    ) -> Dict[str, float]:
        if not BERTSCORE_AVAILABLE:
            raise ImportError("bert-score/transformers is not installed. Install with: pip install bert-score transformers")

        tokenizer = AutoTokenizer.from_pretrained(self.bert_model)

        preds = [truncate_text(x, tokenizer, self.bert_max_tokens) for x in preds]
        gts = [truncate_text(x, tokenizer, self.bert_max_tokens) for x in gts]

        P, R, F1 = bertscore_score(
            preds,
            gts,
            lang="en",
            model_type=self.bert_model,
            verbose=False,
            batch_size=16,
        )

        return {
            "bertscore_precision": float(P.mean().item()),
            "bertscore_recall": float(R.mean().item()),
            "bertscore_f1": float(F1.mean().item()),
        }

    def compute_bertscore_per_sample(
        self,
        preds: List[str],
        gts: List[str],
    ) -> pd.DataFrame:
        if not BERTSCORE_AVAILABLE:
            raise ImportError("bert-score/transformers is not installed. Install with: pip install bert-score transformers")

        tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
        preds = [truncate_text(x, tokenizer, self.bert_max_tokens) for x in preds]
        gts = [truncate_text(x, tokenizer, self.bert_max_tokens) for x in gts]

        P, R, F1 = bertscore_score(
            preds,
            gts,
            lang="en",
            model_type=self.bert_model,
            verbose=False,
            batch_size=16,
        )

        return pd.DataFrame(
            {
                "bertscore_precision": P.tolist(),
                "bertscore_recall": R.tolist(),
                "bertscore_f1": F1.tolist(),
            }
        )

    def compute_mauve(
        self,
        preds: List[str],
        gts: List[str],
    ) -> float:
        """Corpus-level MAUVE.

        MAUVE is not meaningful per-sample. This returns one corpus-level score.
        """
        if not MAUVE_AVAILABLE:
            raise ImportError("mauve-text is not installed. Install with: pip install mauve-text")

        preds = ["" if x is None else str(x) for x in preds]
        gts = ["" if x is None else str(x) for x in gts]

        if self.mauve_max_texts and len(preds) > self.mauve_max_texts:
            preds = preds[: self.mauve_max_texts]
            gts = gts[: self.mauve_max_texts]

        if len(preds) == 0:
            return np.nan

        result = mauve.compute_mauve(
            p_text=preds,
            q_text=gts,
            verbose=False,
        )
        return float(result.mauve)

    # =====================================================
    # Citation/support metrics
    # =====================================================

    def citation_precision_single(
        self,
        answer: Any,
        context: Any,
    ) -> float:
        """Approximate fraction of answer content supported by context."""
        ans_tokens = self._content_token_set(answer)
        ctx_tokens = self._content_token_set(" ".join(safe_parse_context(context)))

        if not ans_tokens:
            return 0.0

        return float(len(ans_tokens & ctx_tokens) / len(ans_tokens))

    def citation_recall_single(
        self,
        answer: Any,
        reference: Any,
        context: Any,
    ) -> float:
        """Approximate recall of gold context-supported content covered by answer."""
        ans_tokens = self._content_token_set(answer)
        ref_tokens = self._content_token_set(reference)
        ctx_tokens = self._content_token_set(" ".join(safe_parse_context(context)))

        support_tokens = ref_tokens & ctx_tokens

        if not support_tokens:
            return 0.0

        return float(len(ans_tokens & support_tokens) / len(support_tokens))

    def compute_citation_metrics_from_dataframe(
        self,
        df: pd.DataFrame,
        answer_col: str = "answer",
        reference_col: str = "gold_ans",
        context_col: str = "context",
    ) -> Dict[str, float]:
        """Corpus-level citation precision/recall.

        Uses existing columns if present; otherwise computes model-free overlap.
        """
        out: Dict[str, float] = {}

        if "citation_precision" in df.columns:
            out["citation_precision"] = float(pd.to_numeric(df["citation_precision"], errors="coerce").mean())
        else:
            if context_col not in df.columns:
                out["citation_precision"] = np.nan
            else:
                out["citation_precision"] = _mean(
                    [
                        self.citation_precision_single(row.get(answer_col, ""), row.get(context_col, ""))
                        for _, row in df.iterrows()
                    ]
                )

        if "citation_recall" in df.columns:
            out["citation_recall"] = float(pd.to_numeric(df["citation_recall"], errors="coerce").mean())
        else:
            if context_col not in df.columns or reference_col not in df.columns:
                out["citation_recall"] = np.nan
            else:
                out["citation_recall"] = _mean(
                    [
                        self.citation_recall_single(
                            row.get(answer_col, ""),
                            row.get(reference_col, ""),
                            row.get(context_col, ""),
                        )
                        for _, row in df.iterrows()
                    ]
                )

        return out

    def compute_citation_metrics_per_sample(
        self,
        df: pd.DataFrame,
        answer_col: str = "answer",
        reference_col: str = "gold_ans",
        context_col: str = "context",
    ) -> pd.DataFrame:
        rows = []

        for _, row in df.iterrows():
            if "citation_precision" in df.columns:
                cp = pd.to_numeric(pd.Series([row.get("citation_precision")]), errors="coerce").iloc[0]
            else:
                cp = self.citation_precision_single(row.get(answer_col, ""), row.get(context_col, ""))

            if "citation_recall" in df.columns:
                cr = pd.to_numeric(pd.Series([row.get("citation_recall")]), errors="coerce").iloc[0]
            else:
                cr = self.citation_recall_single(
                    row.get(answer_col, ""),
                    row.get(reference_col, ""),
                    row.get(context_col, ""),
                )

            rows.append(
                {
                    "citation_precision": float(cp) if not pd.isna(cp) else np.nan,
                    "citation_recall": float(cr) if not pd.isna(cr) else np.nan,
                }
            )

        return pd.DataFrame(rows)

    def _content_token_set(self, text: Any) -> set[str]:
        toks = normalize_text(text).split()
        return {t for t in toks if len(t) >= self.citation_min_token_len}

    # =====================================================
    # Main evaluation
    # =====================================================

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        contexts: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        """Return corpus-level metrics."""
        predictions = ["" if x is None else str(x) for x in predictions]
        references = ["" if x is None else str(x) for x in references]

        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")

        results: Dict[str, float] = {}

        if "bertscore" in self.metrics:
            results.update(self.compute_bertscore(predictions, references))

        if "mauve" in self.metrics:
            results["mauve"] = self.compute_mauve(predictions, references)

        if "rouge" in self.metrics:
            for rouge_metric in ["rouge1", "rouge2", "rougeL"]:
                scores = [
                    self._compute_rouge(pred, gt, rouge_metric)
                    for pred, gt in zip(predictions, references)
                ]
                results[rouge_metric] = _mean(scores)

        for metric in self.metrics:
            if metric in {"bertscore", "mauve", "rouge", "citation", "citation_precision", "citation_recall"}:
                continue

            if metric not in self.available_metrics:
                raise ValueError(f"Unsupported metric: {metric}")

            scores = [
                self.available_metrics[metric](pred, gt)
                for pred, gt in zip(predictions, references)
            ]
            results[metric] = _mean(scores)

        if "citation" in self.metrics or "citation_precision" in self.metrics or "citation_recall" in self.metrics:
            if contexts is None:
                results["citation_precision"] = np.nan
                results["citation_recall"] = np.nan
            else:
                temp_df = pd.DataFrame(
                    {
                        "answer": predictions,
                        "gold_ans": references,
                        "context": contexts,
                    }
                )
                citation = self.compute_citation_metrics_from_dataframe(temp_df)
                if "citation" in self.metrics or "citation_precision" in self.metrics:
                    results["citation_precision"] = citation["citation_precision"]
                if "citation" in self.metrics or "citation_recall" in self.metrics:
                    results["citation_recall"] = citation["citation_recall"]

        return results

    def evaluate_per_sample(
        self,
        predictions: List[str],
        references: List[str],
        contexts: Optional[List[Any]] = None,
    ) -> pd.DataFrame:
        """Return per-sample metrics.

        Corpus-only metrics like MAUVE are repeated as a constant column.
        """
        predictions = ["" if x is None else str(x) for x in predictions]
        references = ["" if x is None else str(x) for x in references]

        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")

        rows = [
            {
                "prediction": pred,
                "reference": gt,
            }
            for pred, gt in zip(predictions, references)
        ]
        metric_df = pd.DataFrame(rows)

        if "bertscore" in self.metrics:
            bert_df = self.compute_bertscore_per_sample(predictions, references)
            metric_df = pd.concat([metric_df, bert_df], axis=1)

        if "mauve" in self.metrics:
            metric_df["mauve"] = self.compute_mauve(predictions, references)

        if "rouge" in self.metrics:
            for rouge_metric in ["rouge1", "rouge2", "rougeL"]:
                metric_df[rouge_metric] = [
                    self._compute_rouge(pred, gt, rouge_metric)
                    for pred, gt in zip(predictions, references)
                ]

        for metric in self.metrics:
            if metric in {"bertscore", "mauve", "rouge", "citation", "citation_precision", "citation_recall"}:
                continue

            if metric not in self.available_metrics:
                raise ValueError(f"Unsupported metric: {metric}")

            metric_df[metric] = [
                self.available_metrics[metric](pred, gt)
                for pred, gt in zip(predictions, references)
            ]

        if "citation" in self.metrics or "citation_precision" in self.metrics or "citation_recall" in self.metrics:
            if contexts is None:
                if "citation" in self.metrics or "citation_precision" in self.metrics:
                    metric_df["citation_precision"] = np.nan
                if "citation" in self.metrics or "citation_recall" in self.metrics:
                    metric_df["citation_recall"] = np.nan
            else:
                temp_df = pd.DataFrame(
                    {
                        "answer": predictions,
                        "gold_ans": references,
                        "context": contexts,
                    }
                )
                citation_df = self.compute_citation_metrics_per_sample(temp_df)
                if "citation" in self.metrics or "citation_precision" in self.metrics:
                    metric_df["citation_precision"] = citation_df["citation_precision"].values
                if "citation" in self.metrics or "citation_recall" in self.metrics:
                    metric_df["citation_recall"] = citation_df["citation_recall"].values

        return metric_df

    # =====================================================
    # DataFrame helpers
    # =====================================================

    def evaluate_from_dataframe(
        self,
        df: pd.DataFrame,
        prediction_col: str = "answer",
        reference_col: str = "gold_ans",
        context_col: Optional[str] = "context",
    ) -> Dict[str, float]:
        predictions = df[prediction_col].fillna("").astype(str).tolist()
        references = df[reference_col].fillna("").astype(str).tolist()

        contexts = None
        if context_col and context_col in df.columns:
            contexts = df[context_col].tolist()

        results = self.evaluate(predictions, references, contexts=contexts)

        if "citation" in self.metrics or "citation_precision" in self.metrics or "citation_recall" in self.metrics:
            citation = self.compute_citation_metrics_from_dataframe(
                df,
                answer_col=prediction_col,
                reference_col=reference_col,
                context_col=context_col or "context",
            )
            results.update(citation)

        return results

    def evaluate_per_sample_from_dataframe(
        self,
        df: pd.DataFrame,
        prediction_col: str = "answer",
        reference_col: str = "gold_ans",
        context_col: Optional[str] = "context",
    ) -> pd.DataFrame:
        predictions = df[prediction_col].fillna("").astype(str).tolist()
        references = df[reference_col].fillna("").astype(str).tolist()

        contexts = None
        if context_col and context_col in df.columns:
            contexts = df[context_col].tolist()

        return self.evaluate_per_sample(predictions, references, contexts=contexts)

    def compare_systems(
        self,
        system1_df: pd.DataFrame,
        system2_df: pd.DataFrame,
        prediction_col: str = "answer",
        reference_col: str = "gold_ans",
        context_col: Optional[str] = "context",
        system1_name: str = "System 1",
        system2_name: str = "System 2",
    ) -> pd.DataFrame:
        res1 = self.evaluate_from_dataframe(
            system1_df,
            prediction_col=prediction_col,
            reference_col=reference_col,
            context_col=context_col,
        )
        res2 = self.evaluate_from_dataframe(
            system2_df,
            prediction_col=prediction_col,
            reference_col=reference_col,
            context_col=context_col,
        )

        rows = []
        for metric in sorted(set(res1) | set(res2)):
            s1 = res1.get(metric, np.nan)
            s2 = res2.get(metric, np.nan)
            rows.append(
                {
                    "metric": metric,
                    system1_name: s1,
                    system2_name: s2,
                    "delta": s2 - s1 if not pd.isna(s1) and not pd.isna(s2) else np.nan,
                }
            )

        return pd.DataFrame(rows)

    def compare_filtered_answer_quality(
        self,
        df: pd.DataFrame,
        accepted_mask: Iterable[bool] | str = "is_accepted",
        prediction_col: str = "answer",
        reference_col: str = "gold_ans",
        context_col: Optional[str] = "context",
    ) -> pd.DataFrame:
        """Compare full dataset vs accepted set after filtering.

        This is the helper expected by RagasPipeline mode 2.
        """
        if isinstance(accepted_mask, str):
            if accepted_mask not in df.columns:
                raise ValueError(f"accepted_mask column not found: {accepted_mask}")
            mask = df[accepted_mask].astype(bool)
        else:
            mask = pd.Series(list(accepted_mask), index=df.index).astype(bool)

        full = df.copy()
        accepted = df.loc[mask].copy()

        full_scores = self.evaluate_from_dataframe(
            full,
            prediction_col=prediction_col,
            reference_col=reference_col,
            context_col=context_col,
        )

        accepted_scores = self.evaluate_from_dataframe(
            accepted,
            prediction_col=prediction_col,
            reference_col=reference_col,
            context_col=context_col,
        ) if len(accepted) > 0 else {k: np.nan for k in full_scores}

        rows = []
        for metric in sorted(set(full_scores) | set(accepted_scores)):
            before = full_scores.get(metric, np.nan)
            after = accepted_scores.get(metric, np.nan)
            rows.append(
                {
                    "metric": metric,
                    "unfiltered": before,
                    "accepted": after,
                    "delta": after - before if not pd.isna(before) and not pd.isna(after) else np.nan,
                }
            )

        summary = pd.DataFrame(rows)
        summary.insert(0, "accepted_samples", int(mask.sum()))
        summary.insert(0, "total_samples", int(len(df)))
        summary.insert(2, "accept_rate", float(mask.mean()) if len(mask) else np.nan)

        return summary


# =========================================================
# Convenience functions
# =========================================================

def evaluate_traditional_metrics(
    df: pd.DataFrame,
    prediction_col: str = "answer",
    reference_col: str = "gold_ans",
    context_col: Optional[str] = "context",
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    evaluator = TraditionalEvaluator(metrics=metrics)
    return evaluator.evaluate_from_dataframe(
        df,
        prediction_col=prediction_col,
        reference_col=reference_col,
        context_col=context_col,
    )


def evaluate_traditional_metrics_per_sample(
    df: pd.DataFrame,
    prediction_col: str = "answer",
    reference_col: str = "gold_ans",
    context_col: Optional[str] = "context",
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    evaluator = TraditionalEvaluator(metrics=metrics)
    return evaluator.evaluate_per_sample_from_dataframe(
        df,
        prediction_col=prediction_col,
        reference_col=reference_col,
        context_col=context_col,
    )


def compare_traditional_systems(
    system1_df: pd.DataFrame,
    system2_df: pd.DataFrame,
    prediction_col: str = "answer",
    reference_col: str = "gold_ans",
    context_col: Optional[str] = "context",
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
        context_col=context_col,
        system1_name=system1_name,
        system2_name=system2_name,
    )


def compare_filtered_answer_quality(
    df: pd.DataFrame,
    accepted_mask: Iterable[bool] | str = "is_accepted",
    prediction_col: str = "answer",
    reference_col: str = "gold_ans",
    context_col: Optional[str] = "context",
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    evaluator = TraditionalEvaluator(metrics=metrics)
    return evaluator.compare_filtered_answer_quality(
        df,
        accepted_mask=accepted_mask,
        prediction_col=prediction_col,
        reference_col=reference_col,
        context_col=context_col,
    )
