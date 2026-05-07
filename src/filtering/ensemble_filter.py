"""
Ensemble answer quality filter combining multiple black-box signals.

Extracts features from (question, answer) pairs using a learned
DeBERTa classifier, an NLI model, and lexical heuristics, then
combines them via a lightweight meta-classifier (logistic regression).

Operates as a black-box: no context or ground truth at inference.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from .data_models import FilterDecision

logger = logging.getLogger(__name__)

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "filtering.yaml"
)


def _load_ensemble_config() -> dict:
    """Load ``ensemble_filter`` section from ``filtering.yaml``."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("ensemble_filter", {})


# ------------------------------------------------------------------
# Feature extraction (pure functions, no model state)
# ------------------------------------------------------------------

def _token_overlap(text_a: str, text_b: str) -> float:
    """Compute token-level F1 between two texts."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    common = tokens_a & tokens_b
    if not common:
        return 0.0
    precision = len(common) / len(tokens_b)
    recall = len(common) / len(tokens_a)
    return 2 * precision * recall / (precision + recall)


def _char_features(answer: str) -> Dict[str, float]:
    """Length and structure features from the answer text."""
    words = answer.split()
    sentences = [
        s.strip() for s in answer.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]
    return {
        "answer_char_len": float(len(answer)),
        "answer_word_count": float(len(words)),
        "answer_sentence_count": float(len(sentences)),
        "answer_avg_word_len": (
            float(np.mean([len(w) for w in words])) if words else 0.0
        ),
    }


@dataclass
class FeatureVector:
    """Named feature vector for a single (question, answer) sample."""

    deberta_confidence: float = 0.0
    nli_entailment: float = 0.0
    qa_token_overlap: float = 0.0
    answer_char_len: float = 0.0
    answer_word_count: float = 0.0
    answer_sentence_count: float = 0.0
    answer_avg_word_len: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.deberta_confidence,
            self.nli_entailment,
            self.qa_token_overlap,
            self.answer_char_len,
            self.answer_word_count,
            self.answer_sentence_count,
            self.answer_avg_word_len,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "deberta_confidence",
            "nli_entailment",
            "qa_token_overlap",
            "answer_char_len",
            "answer_word_count",
            "answer_sentence_count",
            "answer_avg_word_len",
        ]


# ------------------------------------------------------------------
# Ensemble classifier
# ------------------------------------------------------------------

class EnsembleFilter:
    """Meta-classifier combining DeBERTa + NLI + lexical features.

    Usage::

        filt = EnsembleFilter(
            deberta_clf=deberta_clf,
            nli_filter=nli_filter,
        )
        filt.fit(train_questions, train_answers, train_labels)
        decision = filt.predict("question?", "answer text")
    """

    def __init__(
        self,
        deberta_clf: Optional[object] = None,
        nli_filter: Optional[object] = None,
        threshold: float | None = None,
    ) -> None:
        cfg = _load_ensemble_config()
        self.threshold = (
            threshold if threshold is not None
            else cfg.get("threshold", 0.5)
        )
        self.deberta_clf = deberta_clf
        self.nli_filter = nli_filter
        self.scaler = StandardScaler()
        self.meta_clf = LogisticRegression(
            C=cfg.get("regularization_C", 1.0),
            max_iter=cfg.get("max_iter", 1000),
            random_state=cfg.get("seed", 42),
        )
        self._is_fitted = False

    def _extract_features_batch(
        self,
        questions: List[str],
        answers: List[str],
    ) -> np.ndarray:
        """Build the feature matrix for a list of (q, a) pairs."""
        deberta_confs = [0.0] * len(questions)
        if self.deberta_clf is not None:
            decisions = self.deberta_clf.predict_batch(questions, answers)
            deberta_confs = [d.confidence for d in decisions]

        nli_confs = [0.0] * len(questions)
        if self.nli_filter is not None:
            decisions = self.nli_filter.predict_batch(questions, answers)
            nli_confs = [d.confidence for d in decisions]

        rows: List[np.ndarray] = []
        for i, (q, a) in enumerate(zip(questions, answers)):
            char_feats = _char_features(a)
            fv = FeatureVector(
                deberta_confidence=deberta_confs[i],
                nli_entailment=nli_confs[i],
                qa_token_overlap=_token_overlap(q, a),
                **char_feats,
            )
            rows.append(fv.to_array())

        return np.vstack(rows)

    def fit(
        self,
        questions: List[str],
        answers: List[str],
        labels: Sequence[int],
    ) -> Dict[str, float]:
        """Train the meta-classifier on extracted features.

        Returns training metrics dict.
        """
        logger.info("Extracting ensemble features for %d samples...", len(questions))
        X = self._extract_features_batch(questions, answers)
        y = np.array(labels)

        X_scaled = self.scaler.fit_transform(X)
        self.meta_clf.fit(X_scaled, y)
        self._is_fitted = True

        preds = self.meta_clf.predict(X_scaled)
        metrics = {
            "train_accuracy": accuracy_score(y, preds),
            "train_f1": f1_score(y, preds, zero_division=0),
            "train_precision": precision_score(y, preds, zero_division=0),
            "train_recall": recall_score(y, preds, zero_division=0),
        }
        logger.info("Ensemble train metrics: %s", metrics)
        return metrics

    def predict(self, question: str, answer: str) -> FilterDecision:
        """Score a single (question, answer) pair."""
        if not self._is_fitted:
            raise RuntimeError("EnsembleFilter has not been fitted yet.")

        X = self._extract_features_batch([question], [answer])
        X_scaled = self.scaler.transform(X)
        prob = self.meta_clf.predict_proba(X_scaled)[0, 1]

        return FilterDecision(
            accept=prob >= self.threshold,
            confidence=float(prob),
            reasoning=f"ensemble_P(correct)={prob:.3f}, threshold={self.threshold}",
        )

    def predict_batch(
        self,
        questions: List[str],
        answers: List[str],
    ) -> List[FilterDecision]:
        """Score a batch of (question, answer) pairs."""
        if not self._is_fitted:
            raise RuntimeError("EnsembleFilter has not been fitted yet.")

        X = self._extract_features_batch(questions, answers)
        X_scaled = self.scaler.transform(X)
        probs = self.meta_clf.predict_proba(X_scaled)[:, 1]

        decisions: List[FilterDecision] = []
        for prob in probs:
            decisions.append(FilterDecision(
                accept=prob >= self.threshold,
                confidence=float(prob),
                reasoning=(
                    f"ensemble_P(correct)={prob:.3f}, "
                    f"threshold={self.threshold}"
                ),
            ))
        return decisions

    def save(self, path: str | Path) -> None:
        """Persist the fitted scaler and meta-classifier."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "ensemble_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(path / "ensemble_meta_clf.pkl", "wb") as f:
            pickle.dump(self.meta_clf, f)
        logger.info("Ensemble filter saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load a previously fitted scaler and meta-classifier."""
        path = Path(path)
        with open(path / "ensemble_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)  # noqa: S301
        with open(path / "ensemble_meta_clf.pkl", "rb") as f:
            self.meta_clf = pickle.load(f)  # noqa: S301
        self._is_fitted = True
        logger.info("Ensemble filter loaded from %s", path)
