"""
Zero-shot NLI-based answer quality filter.

Uses a pre-trained NLI model to check whether the question
"entails" the answer. No fine-tuning required -- this leverages
the model's existing understanding of textual entailment.

The filter operates as a black-box: ``predict(question, answer)``
returns a ``FilterDecision`` without context or ground truth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data_models import FilterDecision

logger = logging.getLogger(__name__)

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "filtering.yaml"
)


def _load_nli_filter_config() -> dict:
    """Load the ``nli_filter`` section from ``filtering.yaml``."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("nli_filter", {})


class NLIAnswerFilter:
    """Zero-shot answer quality filter using NLI entailment scores.

    Frames answer verification as an NLI problem:
    - premise = question
    - hypothesis = answer
    - entailment score = confidence that the answer is valid

    Usage::

        filt = NLIAnswerFilter()
        decision = filt.predict(
            "When was Python released?",
            "Python 3.0 was released in 2008.",
        )
        print(decision.accept, decision.confidence)
    """

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float | None = None,
        device: str | None = None,
    ) -> None:
        cfg = _load_nli_filter_config()
        self.model_name = model_name or cfg.get(
            "model_name",
            "microsoft/deberta-v3-base-mnli-fever-anli",
        )
        self.threshold = (
            threshold if threshold is not None
            else cfg.get("threshold", 0.5)
        )
        self.max_length: int = cfg.get("max_length", 512)

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
        )
        self.model.to(self.device)
        self.model.eval()

        self._label_map = self._build_label_map()
        logger.info(
            "NLIAnswerFilter loaded: %s (threshold=%.2f, device=%s)",
            self.model_name, self.threshold, self.device,
        )

    def _build_label_map(self) -> Dict[str, int]:
        """Map NLI label names to their indices."""
        id2label = self.model.config.id2label
        label_map: Dict[str, int] = {}
        for idx, name in id2label.items():
            label_map[name.lower()] = int(idx)
        return label_map

    def _get_entailment_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract the entailment probability from NLI logits."""
        probs = torch.softmax(logits, dim=-1)
        ent_idx = self._label_map.get("entailment", 2)
        return probs[:, ent_idx]

    def predict(self, question: str, answer: str) -> FilterDecision:
        """Score a single (question, answer) pair via NLI."""
        inputs = self.tokenizer(
            question, answer,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            prob = self._get_entailment_prob(logits)[0].item()

        return FilterDecision(
            accept=prob >= self.threshold,
            confidence=prob,
            reasoning=(
                f"NLI_entailment={prob:.3f}, "
                f"threshold={self.threshold}"
            ),
        )

    def predict_batch(
        self,
        questions: List[str],
        answers: List[str],
        batch_size: int = 32,
    ) -> List[FilterDecision]:
        """Score a batch of (question, answer) pairs via NLI."""
        decisions: List[FilterDecision] = []

        for start in range(0, len(questions), batch_size):
            batch_q = questions[start: start + batch_size]
            batch_a = answers[start: start + batch_size]

            inputs = self.tokenizer(
                batch_q, batch_a,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = self._get_entailment_prob(logits)
                probs = probs.cpu().tolist()

            for prob in probs:
                decisions.append(FilterDecision(
                    accept=prob >= self.threshold,
                    confidence=prob,
                    reasoning=(
                        f"NLI_entailment={prob:.3f}, "
                        f"threshold={self.threshold}"
                    ),
                ))

        return decisions
