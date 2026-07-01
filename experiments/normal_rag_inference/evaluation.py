"""Evaluation and output diagnostics for normal RAG inference."""

from __future__ import annotations

import re
import statistics
import string
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

_ABSTENTION = re.compile(
    r"\b(?:i\s+don'?t\s+know|cannot\s+answer|can't\s+answer|not\s+provided)\b",
    flags=re.IGNORECASE,
)
_SPECIAL_TOKEN = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<pad>|</s>|\[Retrieval\]|<paragraph>",
    flags=re.IGNORECASE,
)


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation/articles, and normalize whitespace."""

    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: List[str], right: List[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_answer_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute aggregate EM, token-F1, and ROUGE-L."""

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return {"exact_match": 0.0, "token_f1": 0.0, "rouge_l": 0.0}

    count = len(predictions)
    return {
        "exact_match": sum(exact_match(p, r) for p, r in zip(predictions, references)) / count,
        "token_f1": sum(token_f1(p, r) for p, r in zip(predictions, references)) / count,
        "rouge_l": sum(rouge_l(p, r) for p, r in zip(predictions, references)) / count,
    }


def compute_output_diagnostics(predictions: Iterable[str]) -> Dict[str, float]:
    """Compute lightweight diagnostics that catch malformed generation outputs."""

    cleaned = [str(prediction).strip() for prediction in predictions]
    word_counts = [len(prediction.split()) for prediction in cleaned]
    return {
        "empty_predictions": sum(1 for prediction in cleaned if not prediction),
        "abstention_predictions": sum(
            1 for prediction in cleaned if _ABSTENTION.search(prediction)
        ),
        "special_token_leaks": sum(
            1 for prediction in cleaned if _SPECIAL_TOKEN.search(prediction)
        ),
        "avg_prediction_words": statistics.mean(word_counts) if word_counts else 0.0,
        "median_prediction_words": statistics.median(word_counts) if word_counts else 0.0,
    }


def _metrics_for_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    predictions = [str(row["predicted_answer"]) for row in rows]
    references = [str(row["gold_answer"]) for row in rows]
    metrics = compute_answer_metrics(predictions=predictions, references=references)
    metrics["n_samples"] = len(rows)
    return metrics


def compute_grouped_metrics(
    rows: List[Dict[str, Any]],
    group_fields: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute metrics by individual fields and by dataset+label when present."""

    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for field in group_fields:
            if field in row:
                grouped[field][str(row[field])].append(row)
        if "dataset" in row and "label" in row:
            grouped["dataset_label"][f"{row['dataset']}|{row['label']}"].append(row)

    return {
        field: {value: _metrics_for_rows(value_rows) for value, value_rows in groups.items()}
        for field, groups in grouped.items()
    }
