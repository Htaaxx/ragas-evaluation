"""
Learned answer quality classifier (core thesis component).

Provides:
- ``AnswerQualityClassifier`` — inference-time accept/reject filter
  that takes (question, answer) and returns a ``FilterDecision``
  without requiring ground truth or context.
- ``train_classifier()`` — fine-tunes DeBERTa-v3-small on
  ``labeled_asqa.csv`` using HuggingFace Trainer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .data_models import FilterDecision

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "filtering.yaml"


def _load_learned_filter_config() -> dict:
    """Load the ``learned_filter`` section from ``filtering.yaml``."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("learned_filter", {})


# ---------------------------------------------------------------------------
# HuggingFace Dataset wrapper
# ---------------------------------------------------------------------------

class _QADataset(Dataset):
    """Tokenised (question, answer) pairs with binary labels."""

    def __init__(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        labels: Sequence[int],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None:
        self.encodings = tokenizer(
            list(questions),
            list(answers),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class AnswerQualityClassifier:
    """Learned answer-only filter. No ground truth or context needed.

    Usage::

        clf = AnswerQualityClassifier("models/answer_filter")
        decision = clf.predict("When was Python released?", "Python 3.0 was released in 2008.")
        print(decision.accept, decision.confidence)
    """

    def __init__(
        self,
        model_path: str,
        threshold: float | None = None,
        device: str | None = None,
    ) -> None:
        cfg = _load_learned_filter_config()
        self.threshold = threshold if threshold is not None else cfg.get("threshold", 0.5)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.max_length: int = cfg.get("max_length", 512)
        logger.info(
            "AnswerQualityClassifier loaded from %s (threshold=%.2f, device=%s)",
            model_path, self.threshold, self.device,
        )

    def predict(self, question: str, answer: str) -> FilterDecision:
        """Score a single (question, answer) pair."""
        inputs = self.tokenizer(
            question, answer,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            prob_correct = torch.softmax(logits, dim=-1)[0, 1].item()

        return FilterDecision(
            accept=prob_correct >= self.threshold,
            confidence=prob_correct,
            reasoning=f"P(correct)={prob_correct:.3f}, threshold={self.threshold}",
        )

    def predict_batch(
        self,
        questions: List[str],
        answers: List[str],
        batch_size: int = 32,
    ) -> List[FilterDecision]:
        """Score a batch of (question, answer) pairs."""
        decisions: List[FilterDecision] = []

        for start in range(0, len(questions), batch_size):
            batch_q = questions[start : start + batch_size]
            batch_a = answers[start : start + batch_size]

            inputs = self.tokenizer(
                batch_q, batch_a,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()

            for prob in probs:
                decisions.append(FilterDecision(
                    accept=prob >= self.threshold,
                    confidence=prob,
                    reasoning=f"P(correct)={prob:.3f}, threshold={self.threshold}",
                ))

        return decisions


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _compute_metrics(eval_pred) -> Dict[str, float]:
    """Metric callback for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }


def train_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str | None = None,
    output_dir: str | None = None,
    config_overrides: dict | None = None,
) -> Path:
    """Fine-tune a DeBERTa classifier on labeled answer-quality data.

    Parameters
    ----------
    train_df / val_df:
        DataFrames with columns ``question``, ``answer``, ``label``.
    model_name:
        HuggingFace model ID. Defaults to ``filtering.yaml`` value.
    output_dir:
        Where to save the final model. Defaults to ``filtering.yaml`` value.
    config_overrides:
        Optional dict to override any ``learned_filter`` config values.

    Returns
    -------
    Path to the saved model directory.
    """
    cfg = _load_learned_filter_config()
    if config_overrides:
        cfg.update(config_overrides)

    model_name = model_name or cfg["model_name"]
    output_dir = output_dir or cfg["model_path"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    max_length: int = cfg.get("max_length", 512)
    batch_size: int = cfg.get("batch_size", 16)
    num_epochs: int = cfg.get("num_epochs", 3)
    learning_rate: float = cfg.get("learning_rate", 2e-5)
    warmup_ratio: float = cfg.get("warmup_ratio", 0.1)
    weight_decay: float = cfg.get("weight_decay", 0.01)
    seed: int = cfg.get("seed", 42)

    logger.info("Training config: %s", json.dumps(cfg, indent=2))
    logger.info("Model: %s  ->  %s", model_name, output_path)
    logger.info("Train samples: %d  Val samples: %d", len(train_df), len(val_df))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
    )

    train_dataset = _QADataset(
        train_df["question"].tolist(),
        train_df["answer"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        max_length=max_length,
    )
    val_dataset = _QADataset(
        val_df["question"].tolist(),
        val_df["answer"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        max_length=max_length,
    )

    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=seed,
        fp16=False, # DeBERTa-v3-small is small enough to train on CPU if needed
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training …")
    train_result = trainer.train()

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    logger.info("Model saved to %s", output_path)

    training_log = {
        "config": cfg,
        "model_name": model_name,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "train_metrics": {
            k: round(v, 6) for k, v in train_result.metrics.items()
        },
    }

    eval_metrics = trainer.evaluate()
    training_log["val_metrics"] = {
        k: round(v, 6) for k, v in eval_metrics.items()
    }

    log_path = Path("results") / "training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(training_log, fh, indent=2)
    logger.info("Training log saved to %s", log_path)

    return output_path
