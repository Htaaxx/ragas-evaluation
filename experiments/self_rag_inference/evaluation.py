"""Answer-generation metrics for Self-RAG inference."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation/articles, and normalize whitespace."""

    text = text.lower()
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
