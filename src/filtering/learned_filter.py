"""
Learned answer quality classifier (core thesis component).

Provides:
- ``AnswerQualityClassifier`` — inference-time faithfulness filter
  that takes (context, answer) and returns a ``FilterDecision``.
  Retrieval is assumed correct; the task is verifying whether the
  generated answer is faithful to the retrieved context.
- ``train_classifier()`` — fine-tunes DeBERTa on
  ``labeled_asqa.csv`` using HuggingFace Trainer with NLI-style
  framing: premise=context, hypothesis=answer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .data_models import FilterDecision

logger = logging.getLogger(__name__)

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "filtering.yaml"
)


def _load_learned_filter_config() -> dict:
    """Load the ``learned_filter`` section from ``filtering.yaml``."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("learned_filter", {})


def _fix_deberta_layernorm_keys(model) -> None:
    """Replace DebertaV2LayerNorm with standard nn.LayerNorm in-place.

    DebertaV2LayerNorm uses .gamma/.beta naming that causes checkpoint
    save/reload mismatches with HuggingFace Trainer. Replacing with
    standard nn.LayerNorm (which uses .weight/.bias) fixes this entirely.
    """
    import torch.nn as nn

    replacements = []
    for name, module in model.named_modules():
        class_name = type(module).__name__
        if "LayerNorm" in class_name and class_name != "LayerNorm":
            # Get weight data from whichever attribute exists
            if hasattr(module, "gamma") and module.gamma is not None:
                weight_data = module.gamma.data.clone()
            elif hasattr(module, "weight") and module.weight is not None:
                weight_data = module.weight.data.clone()
            else:
                continue

            if hasattr(module, "beta") and module.beta is not None:
                bias_data = module.beta.data.clone()
            elif hasattr(module, "bias") and module.bias is not None:
                bias_data = module.bias.data.clone()
            else:
                bias_data = torch.zeros_like(weight_data)

            eps = getattr(module, "variance_epsilon",
                          getattr(module, "eps", 1e-12))
            replacements.append((name, weight_data, bias_data, eps))

    # Perform replacements (use eps=1e-7 minimum for numerical stability)
    for name, weight_data, bias_data, eps in replacements:
        safe_eps = max(eps, 1e-7)
        new_ln = nn.LayerNorm(weight_data.shape[0], eps=safe_eps)
        new_ln.weight.data.copy_(weight_data)
        new_ln.bias.data.copy_(bias_data)

        # Navigate to parent and replace
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_ln)

    if replacements:
        logger.info(
            "Replaced %d DebertaV2LayerNorm modules with nn.LayerNorm "
            "(guarantees .weight/.bias naming in state_dict)",
            len(replacements),
        )

    # Final verification
    bad_keys = [k for k in model.state_dict().keys()
                if k.endswith(".gamma") or k.endswith(".beta")]
    if bad_keys:
        logger.error(
            "STILL found %d .gamma/.beta keys after fix: %s",
            len(bad_keys), bad_keys[:4],
        )
    else:
        logger.info("Verified: state_dict has NO .gamma/.beta keys")


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

def _extract_top1_context(context_str: str, max_chars: int = 800) -> str:
    """Extract top-1 passage from the context dict string.

    The context column in labeled_asqa.csv is a stringified dict with
    'title' (list) and 'sentences' (list of list of strings). The first
    title+sentences is the primary source passage.
    """
    import ast

    try:
        ctx = ast.literal_eval(context_str)
        title = ctx["title"][0]
        sentences = " ".join(ctx["sentences"][0])
        passage = f"{title}: {sentences}"
        return passage[:max_chars]
    except (ValueError, KeyError, IndexError, TypeError):
        return ""


# ---------------------------------------------------------------------------
# HuggingFace Dataset wrapper (lazy tokenization for dynamic padding)
# ---------------------------------------------------------------------------

class _QADataset(Dataset):
    """Faithfulness verification dataset: (context, answer) pairs.

    NLI-style framing (retrieval assumed correct):
      text_a = context  (premise — the ground truth)
      text_b = answer   (hypothesis — to verify)

    The model learns whether the answer is faithful to the context.
    """

    def __init__(
        self,
        contexts: Sequence[str],
        answers: Sequence[str],
        labels: Sequence[int],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None:
        self.contexts = [str(c) if c is not None else "" for c in contexts]
        self.answers = [str(a) if a is not None else "" for a in answers]
        self.labels = [int(lbl) for lbl in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

        assert all(lbl in (0, 1) for lbl in self.labels), \
            f"Labels must be 0 or 1, got unique: {set(self.labels)}"

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.contexts[idx],
            self.answers[idx],
            truncation=True,
            max_length=self.max_length,
        )
        encoding["labels"] = self.labels[idx]
        return encoding


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class AnswerQualityClassifier:
    """Learned faithfulness filter: context (premise) vs answer (hypothesis).

    Retrieval is assumed correct. The model checks whether the generated
    answer is faithful to the retrieved context.

    Usage::

        clf = AnswerQualityClassifier("models/answer_filter")
        decision = clf.predict(
            context="Python: Python 3.0 was released on December 3, 2008.",
            answer="Python 3.0 was released in 2008.",
        )
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
        _fix_deberta_layernorm_keys(self.model)
        self.model.to(self.device)
        self.model.eval()

        self.max_length: int = cfg.get("max_length", 512)
        logger.info(
            "AnswerQualityClassifier loaded from %s (threshold=%.2f, device=%s)",
            model_path, self.threshold, self.device,
        )

    def predict(
        self,
        context: str,
        answer: str,
    ) -> FilterDecision:
        """Check if answer is faithful to context (NLI framing)."""
        inputs = self.tokenizer(
            context, answer,
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
            reasoning=f"P(faithful)={prob_correct:.3f}, threshold={self.threshold}",
        )

    def predict_batch(
        self,
        contexts: List[str],
        answers: List[str],
        batch_size: int = 32,
    ) -> List[FilterDecision]:
        """Check faithfulness for a batch of (context, answer) pairs."""
        decisions: List[FilterDecision] = []

        for start in range(0, len(contexts), batch_size):
            batch_c = contexts[start : start + batch_size]
            batch_a = answers[start : start + batch_size]

            inputs = self.tokenizer(
                batch_c, batch_a,
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
                    reasoning=f"P(faithful)={prob:.3f}, threshold={self.threshold}",
                ))

        return decisions


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _compute_metrics(eval_pred) -> Dict[str, float]:
    """Metric callback for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Log prediction distribution for debugging
    pred_counts = {0: int((preds == 0).sum()), 1: int((preds == 1).sum())}
    label_counts = {0: int((labels == 0).sum()), 1: int((labels == 1).sum())}
    logits_mean = logits.mean(axis=0).tolist()
    logits_std = logits.std(axis=0).tolist()
    logger.info(
        "EVAL: preds=%s labels=%s logits_mean=%s logits_std=%s",
        pred_counts, label_counts,
        [f"{x:.4f}" for x in logits_mean],
        [f"{x:.4f}" for x in logits_std],
    )

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
    """Fine-tune DeBERTa as a faithfulness classifier (NLI framing).

    Retrieval is assumed correct. The model learns to verify whether
    the generated answer is faithful to the retrieved context:
      premise  = context (ground truth)
      hypothesis = answer (to verify)

    Parameters
    ----------
    train_df / val_df:
        DataFrames with columns ``question``, ``answer``, ``context``,
        ``label``.  The ``context`` column is required (raw string from
        labeled_asqa.csv).
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
    num_epochs: int = cfg.get("num_epochs", 10)
    learning_rate: float = cfg.get("learning_rate", 2e-5)
    warmup_ratio: float = cfg.get("warmup_ratio", 0.1)
    weight_decay: float = cfg.get("weight_decay", 0.01)
    max_grad_norm: float = cfg.get("max_grad_norm", 1.0)
    label_smoothing: float = cfg.get("label_smoothing", 0.0)
    early_stopping_patience: int = cfg.get("early_stopping_patience", 3)
    use_fp16: bool = cfg.get("fp16", False)
    save_total_limit: int = cfg.get("save_total_limit", 3)
    seed: int = cfg.get("seed", 42)

    # --- Context extraction (required) ---
    if "context" not in train_df.columns:
        raise ValueError(
            "train_df must have a 'context' column. "
            "Retrieval is assumed correct; context is the premise."
        )

    logger.info("Extracting top-1 context (premise) for faithfulness training")
    train_contexts = [
        _extract_top1_context(c) for c in train_df["context"].tolist()
    ]
    val_contexts = [
        _extract_top1_context(c) for c in val_df["context"].tolist()
    ]
    non_empty_train = sum(1 for c in train_contexts if c)
    non_empty_val = sum(1 for c in val_contexts if c)
    logger.info(
        "Context extracted: train=%d/%d, val=%d/%d non-empty",
        non_empty_train, len(train_contexts),
        non_empty_val, len(val_contexts),
    )

    # --- Pre-training diagnostics ---
    for name, split in [("train", train_df), ("val", val_df)]:
        nan_labels = split["label"].isna().sum()
        nan_a = split["answer"].isna().sum()
        label_dist = split["label"].value_counts().to_dict()
        logger.info(
            "DIAGNOSTIC [%s]: labels=%s, NaN(label=%d, answer=%d), "
            "label_dtype=%s",
            name, label_dist, nan_labels, nan_a, split["label"].dtype,
        )
        if nan_labels > 0:
            raise ValueError(f"{name} has {nan_labels} NaN labels!")

    logger.info("Training config: %s", json.dumps(cfg, indent=2))
    logger.info("Model: %s  ->  %s", model_name, output_path)
    logger.info(
        "Train: %d samples, Val: %d samples, Framing: context→answer (NLI)",
        len(train_df), len(val_df),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True,
    )
    _fix_deberta_layernorm_keys(model)

    train_dataset = _QADataset(
        train_contexts,
        train_df["answer"].tolist(),
        train_df["label"].astype(int).tolist(),
        tokenizer,
        max_length=max_length,
    )
    val_dataset = _QADataset(
        val_contexts,
        val_df["answer"].tolist(),
        val_df["label"].astype(int).tolist(),
        tokenizer,
        max_length=max_length,
    )

    # Sanity check: verify samples show context→answer framing
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        decoded = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
        logger.info(
            "DIAGNOSTIC sample[%d]: tokens=%d, label=%d, "
            "preview='%s'",
            i, len(sample["input_ids"]), sample["labels"],
            decoded[:200],
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "DIAGNOSTIC params: trainable=%d/%d (%.1f%%)",
        trainable, total, 100 * trainable / total,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
    )

    # Differential learning rates: higher for classifier head, lower for encoder
    classifier_lr: float = cfg.get("classifier_lr", 1e-3)
    classifier_params = []
    encoder_params = []
    for name, param in model.named_parameters():
        if "classifier" in name or "pooler" in name:
            classifier_params.append(param)
        else:
            encoder_params.append(param)

    logger.info(
        "Optimizer: encoder_lr=%.1e (%d params), classifier_lr=%.1e (%d params)",
        learning_rate, len(encoder_params),
        classifier_lr, len(classifier_params),
    )

    from torch.optim import AdamW
    optimizer = AdamW([
        {"params": encoder_params, "lr": learning_rate, "weight_decay": weight_decay},
        {"params": classifier_params, "lr": classifier_lr, "weight_decay": 0.0},
    ])

    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=max(1, 16 // batch_size),
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        label_smoothing_factor=label_smoothing,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=save_total_limit,
        seed=seed,
        fp16=use_fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
            ),
        ],
    )

    # Pre-flight forward pass to catch NaN before wasting time
    logger.info("Running pre-flight forward pass...")
    model.eval()
    with torch.no_grad():
        batch = data_collator([train_dataset[i] for i in range(min(4, len(train_dataset)))])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        logger.info(
            "DIAGNOSTIC pre-flight: loss=%.4f, logits_mean=%.4f, "
            "logits_std=%.4f, any_nan=%s",
            outputs.loss.item(),
            outputs.logits.mean().item(),
            outputs.logits.std().item(),
            bool(torch.isnan(outputs.logits).any()),
        )
        if torch.isnan(outputs.logits).any():
            raise RuntimeError(
                "Model produces NaN logits BEFORE training! "
                "The model weights are corrupted or incompatible."
            )
    model.train()

    logger.info("Starting training ...")
    train_result = trainer.train()

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    logger.info("Model saved to %s", output_path)

    training_log = {
        "config": cfg,
        "model_name": model_name,
        "framing": "context→answer (NLI faithfulness)",
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "train_metrics": {
            k: round(v, 6)
            for k, v in train_result.metrics.items()
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
