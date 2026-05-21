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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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


def _load_deberta_model(model_name_or_path: str, num_labels: int = 2):
    """Load a DeBERTa model with robust LayerNorm key handling.

    Problem: some DeBERTa checkpoints store LayerNorm parameters as
    ``.gamma/.beta`` while the model architecture expects ``.weight/.bias``.
    ``from_pretrained`` silently drops these, leaving LayerNorm at defaults.

    Fix: after ``from_pretrained``, we download the raw checkpoint, find
    any ``.gamma/.beta`` keys, and copy them into the model's
    ``.weight/.bias`` slots. No module replacement — the architecture
    stays exactly as HuggingFace built it, so save/reload always works.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    # --- Remap .gamma/.beta → .weight/.bias from raw checkpoint ---
    raw_state = None
    try:
        model_path = Path(model_name_or_path)
        if model_path.is_dir():
            safe_path = model_path / "model.safetensors"
            pt_path = model_path / "pytorch_model.bin"
            if safe_path.exists():
                from safetensors.torch import load_file
                raw_state = load_file(str(safe_path))
            elif pt_path.exists():
                raw_state = torch.load(
                    str(pt_path), map_location="cpu", weights_only=True,
                )
        else:
            from huggingface_hub import hf_hub_download
            try:
                wf = hf_hub_download(model_name_or_path, "model.safetensors")
                from safetensors.torch import load_file
                raw_state = load_file(wf)
            except Exception:
                wf = hf_hub_download(model_name_or_path, "pytorch_model.bin")
                raw_state = torch.load(wf, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.warning("Could not load raw checkpoint for key fix: %s", exc)

    if raw_state is not None:
        model_state = model.state_dict()
        fixed = 0
        for key, value in raw_state.items():
            if key.endswith(".gamma"):
                target = key[:-6] + ".weight"
            elif key.endswith(".beta"):
                target = key[:-5] + ".bias"
            else:
                continue
            if target in model_state and model_state[target].shape == value.shape:
                model_state[target] = value
                fixed += 1

        if fixed > 0:
            model.load_state_dict(model_state, strict=False)
            logger.info(
                "LayerNorm fix: copied %d .gamma/.beta → .weight/.bias "
                "from raw checkpoint", fixed,
            )
        else:
            logger.info("LayerNorm keys: no .gamma/.beta remapping needed")

    return model


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
# NLI formatting (Step 4) — explicit Premise/Hypothesis role cues
# ---------------------------------------------------------------------------

def _format_premise(context: str) -> str:
    """Wrap context with an explicit Premise: cue for the MNLI head."""
    return f"Premise: {context}"


def _format_hypothesis(answer: str) -> str:
    """Wrap answer with an explicit Hypothesis: cue for the MNLI head."""
    return f"Hypothesis: {answer}"


# Truncation strategy used everywhere (training + inference + diagnostic).
# ``"only_first"`` truncates ONLY the context (text_a = Premise), preserving
# the entire answer (text_b = Hypothesis). Hallucinations differ from correct
# answers in a single trailing entity at the end of the answer, so the answer
# tail MUST never be truncated.
_TRUNCATION_STRATEGY = "only_first"


# ---------------------------------------------------------------------------
# Truncation collision diagnostic (Step 3)
# ---------------------------------------------------------------------------

def _truncation_collision_diagnostic(
    tokenizer,
    df: pd.DataFrame,
    contexts: Sequence[str],
    max_length: int,
    n: int = 200,
    tail_tokens: int = 10,
) -> float:
    """Log the fraction of (correct, hallucinated) pairs whose tokenized
    inputs are identical in their last ``tail_tokens`` after truncation.

    Hallucinations differ from correct answers only in a trailing entity.
    If truncation kills that entity, the model has no signal to learn from.

    Returns the collision rate (0.0 to 1.0).
    """
    # Build a base-id index
    base_col = df["id"].str.replace(r"b$", "", regex=True)
    df_indexed = df.copy()
    df_indexed["_base"] = base_col
    df_indexed["_pos"] = list(range(len(df_indexed)))

    pair_rows = []
    for base, group in df_indexed.groupby("_base"):
        if len(group) < 2:
            continue
        pos_row = group[group["label"] == 1]
        neg_row = group[group["label"] == 0]
        if len(pos_row) == 0 or len(neg_row) == 0:
            continue
        pair_rows.append((int(pos_row.iloc[0]["_pos"]), int(neg_row.iloc[0]["_pos"])))
        if len(pair_rows) >= n:
            break

    if not pair_rows:
        logger.warning(
            "Truncation diagnostic: no paired (asqa_X, asqa_Xb) found"
        )
        return 0.0

    collisions = 0
    for pos_idx, neg_idx in pair_rows:
        context = contexts[pos_idx]
        ans_correct = str(df.iloc[pos_idx]["answer"])
        ans_halluc = str(df.iloc[neg_idx]["answer"])

        toks_c = tokenizer(
            _format_premise(context), _format_hypothesis(ans_correct),
            truncation=_TRUNCATION_STRATEGY, max_length=max_length,
        )["input_ids"]
        toks_h = tokenizer(
            _format_premise(context), _format_hypothesis(ans_halluc),
            truncation=_TRUNCATION_STRATEGY, max_length=max_length,
        )["input_ids"]

        if toks_c[-tail_tokens:] == toks_h[-tail_tokens:]:
            collisions += 1

    rate = collisions / len(pair_rows)
    logger.info(
        "Truncation collision diagnostic: %d/%d pairs (%.1f%%) have "
        "identical last %d tokens after truncation "
        "(max_length=%d, strategy=%s)",
        collisions, len(pair_rows), 100 * rate, tail_tokens,
        max_length, _TRUNCATION_STRATEGY,
    )
    if rate > 0.05:
        logger.warning(
            "Truncation collision rate %.1f%% > 5%%. The discriminative "
            "trailing entity is being lost. Raise max_length "
            "(currently %d) until this drops below 5%%.",
            100 * rate, max_length,
        )
    return rate


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
            _format_premise(self.contexts[idx]),
            _format_hypothesis(self.answers[idx]),
            truncation=_TRUNCATION_STRATEGY,
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
        self.model = _load_deberta_model(model_path, num_labels=2)
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
        logger.info("AnswerQualityClassifier.predict using threshold=%.3f", self.threshold)
        inputs = self.tokenizer(
            _format_premise(context), _format_hypothesis(answer),
            return_tensors="pt",
            truncation=_TRUNCATION_STRATEGY,
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
        logger.info(
            "AnswerQualityClassifier.predict_batch using threshold=%.3f (n=%d)",
            self.threshold, len(contexts),
        )
        decisions: List[FilterDecision] = []

        for start in range(0, len(contexts), batch_size):
            batch_c = [_format_premise(c) for c in contexts[start : start + batch_size]]
            batch_a = [_format_hypothesis(a) for a in answers[start : start + batch_size]]

            inputs = self.tokenizer(
                batch_c, batch_a,
                return_tensors="pt",
                truncation=_TRUNCATION_STRATEGY,
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
    """Metric callback for HuggingFace Trainer.

    Reports the full thesis metric set:
      - confusion matrix counts (tp, tn, fp, fn)
      - precision, recall, f1 at threshold=0.5 (legacy debug)
      - fpr (PRIMARY thesis metric — false positive rate)
      - tpr, accuracy
      - roc_auc (threshold-independent; used for checkpoint selection)
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(labels, probs))
    except ValueError:
        roc_auc = 0.5

    logits_mean = logits.mean(axis=0).tolist()
    logits_std = logits.std(axis=0).tolist()
    logger.info(
        "EVAL: cm=[[TN=%d, FP=%d],[FN=%d, TP=%d]] "
        "fpr=%.3f tpr=%.3f roc_auc=%.3f "
        "logits_mean=%s logits_std=%s",
        tn, fp, fn, tp, fpr, tpr, roc_auc,
        [f"{x:.4f}" for x in logits_mean],
        [f"{x:.4f}" for x in logits_std],
    )

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ---------------------------------------------------------------------------
# Weighted-loss Trainer (Step 11) — optional FP-penalty
# ---------------------------------------------------------------------------

class _WeightedTrainer(Trainer):
    """Trainer that applies per-class CrossEntropy weights.

    Enabled via ``use_weighted_loss: true`` in filtering.yaml. Used as a
    decision-boundary nudge AFTER threshold tuning still cannot reach the
    FPR target. Default weights penalize FP harder (weight on class 0 > 1).
    """

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = (
            self._class_weights.to(logits.device)
            if self._class_weights is not None else None
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fn(logits, labels)
        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Overfit sanity check (Step 5) — the HARD GATE before full training
# ---------------------------------------------------------------------------

def overfit_sanity_check(
    df: pd.DataFrame,
    n_pairs: int = 16,
    epochs: int = 50,
    model_name: str | None = None,
    learning_rate: float = 5e-5,
    max_length: int | None = None,
) -> Dict[str, float]:
    """Try to overfit ``n_pairs`` paired (correct, halluc) examples.

    A 184M-parameter model MUST be able to fit a tiny labelled set. If
    train F1 does not reach >= 0.95 within ``epochs``, the code, data, or
    input format is broken — do NOT scale up to full training.

    Parameters
    ----------
    df:
        DataFrame with ``id``, ``question``, ``answer``, ``context``,
        ``label`` columns.
    n_pairs:
        Number of paired base IDs to use (total samples = 2 * n_pairs).
    epochs:
        Training epochs on the tiny set.
    model_name:
        HuggingFace model ID; defaults to filtering.yaml value.
    learning_rate:
        LR for the overfit run. Higher than full-train default since the
        set is tiny.
    max_length:
        Defaults to filtering.yaml value.

    Returns
    -------
    Dict with ``train_f1``, ``train_loss``, ``train_fpr``,
    ``train_recall``, ``train_precision``, plus per-pair logits for
    debugging when the test fails.
    """
    cfg = _load_learned_filter_config()
    model_name = model_name or cfg["model_name"]
    max_length = max_length or cfg.get("max_length", 384)

    base_col = df["id"].str.replace(r"b$", "", regex=True)
    df_idx = df.copy()
    df_idx["_base"] = base_col

    selected_rows = []
    seen_bases = set()
    for _, row in df_idx.iterrows():
        b = row["_base"]
        if b in seen_bases:
            continue
        group = df_idx[df_idx["_base"] == b]
        if (group["label"] == 1).any() and (group["label"] == 0).any():
            selected_rows.append(group[group["label"] == 1].iloc[0])
            selected_rows.append(group[group["label"] == 0].iloc[0])
            seen_bases.add(b)
        if len(seen_bases) >= n_pairs:
            break

    if len(seen_bases) < n_pairs:
        logger.warning(
            "overfit_sanity_check: only found %d pairs (requested %d)",
            len(seen_bases), n_pairs,
        )

    mini_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    contexts = [_extract_top1_context(c) for c in mini_df["context"].tolist()]
    answers = [str(a) for a in mini_df["answer"].tolist()]
    labels = [int(lbl) for lbl in mini_df["label"].tolist()]
    ids = mini_df["id"].tolist()

    logger.info(
        "OVERFIT TEST: %d samples (%d pos, %d neg) for %d epochs at lr=%.0e",
        len(mini_df),
        sum(1 for lbl in labels if lbl == 1),
        sum(1 for lbl in labels if lbl == 0),
        epochs, learning_rate,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = _load_deberta_model(model_name, num_labels=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    encodings = [
        tokenizer(
            _format_premise(c), _format_hypothesis(a),
            truncation=_TRUNCATION_STRATEGY, max_length=max_length,
            return_tensors="pt",
        )
        for c, a in zip(contexts, answers)
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        # Iterate over batches of 8 (matches full training shape)
        for start in range(0, len(encodings), 8):
            batch = encodings[start : start + 8]
            batch_labels = torch.tensor(
                labels[start : start + 8], device=device,
            )
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [enc["input_ids"].squeeze(0) for enc in batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id or 0,
            ).to(device)
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [enc["attention_mask"].squeeze(0) for enc in batch],
                batch_first=True,
                padding_value=0,
            ).to(device)

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(out.logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "  epoch %d/%d: loss=%.4f",
                epoch + 1, epochs, epoch_loss / max(1, len(encodings) // 8),
            )

    # Final evaluation on training set
    model.eval()
    all_preds: List[int] = []
    all_probs: List[float] = []
    with torch.no_grad():
        for enc in encodings:
            enc_dev = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc_dev)
            probs = torch.softmax(out.logits, dim=-1)[0]
            all_probs.append(probs[1].item())
            all_preds.append(int(probs.argmax().item()))

    cm = confusion_matrix(labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    f1 = f1_score(labels, all_preds, zero_division=0)
    precision = precision_score(labels, all_preds, zero_division=0)
    recall = recall_score(labels, all_preds, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    logger.info(
        "OVERFIT RESULT: F1=%.3f P=%.3f R=%.3f FPR=%.3f "
        "[TN=%d FP=%d FN=%d TP=%d]",
        f1, precision, recall, fpr, tn, fp, fn, tp,
    )

    # If it failed, log the worst-classified pairs for debugging
    if f1 < 0.95:
        logger.error(
            "OVERFIT FAILED — F1=%.3f < 0.95. The code/data/format is broken.",
            f1,
        )
        errors = [
            (ids[i], labels[i], all_preds[i], all_probs[i], answers[i][:120])
            for i in range(len(labels)) if labels[i] != all_preds[i]
        ]
        for sid, lbl, pred, prob, ans in errors[:5]:
            logger.error(
                "  WORST: id=%s label=%d pred=%d p_faithful=%.3f answer='%s...'",
                sid, lbl, pred, prob, ans,
            )

    return {
        "train_f1": float(f1),
        "train_precision": float(precision),
        "train_recall": float(recall),
        "train_fpr": float(fpr),
        "n_samples": len(labels),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
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
    use_weighted_loss: bool = cfg.get("use_weighted_loss", False)
    loss_weight_neg: float = cfg.get("loss_weight_neg", 2.0)
    loss_weight_pos: float = cfg.get("loss_weight_pos", 1.0)

    # Step 8: FP32 enforcement. Filter training must NEVER use fp16
    # (DeBERTa-v3 disentangled attention instability).
    assert use_fp16 is False, (
        "Thesis run must use FP32 for numerical stability. "
        "Set fp16: false in filtering.yaml."
    )
    logger.info("FP32 mode enforced (fp16=False)")

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
        "Train: %d samples, Val: %d samples, Framing: context→answer (NLI), "
        "max_length=%d, truncation=%s (preserves full answer, truncates context)",
        len(train_df), len(val_df), max_length, _TRUNCATION_STRATEGY,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = _load_deberta_model(model_name, num_labels=2)

    # Step 3: truncation collision diagnostic
    _truncation_collision_diagnostic(
        tokenizer, train_df, train_contexts, max_length=max_length, n=200,
    )

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
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        save_total_limit=save_total_limit,
        seed=seed,
        report_to="none",
    )

    if use_weighted_loss:
        class_weights = torch.tensor(
            [loss_weight_neg, loss_weight_pos], dtype=torch.float32,
        )
        logger.info(
            "Using WeightedTrainer (class_weights=[neg=%.2f, pos=%.2f]) "
            "— this nudges the decision boundary to penalize FP",
            loss_weight_neg, loss_weight_pos,
        )
        trainer = _WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
            class_weights=class_weights,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                ),
            ],
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                ),
            ],
        )

    # Pre-flight: test forward on longer sequences to catch NaN early
    logger.info("Running pre-flight on 16 random samples...")
    import random as _rnd
    _rnd.seed(seed)
    pilot_indices = _rnd.sample(range(len(train_dataset)), min(16, len(train_dataset)))
    pilot_batch = data_collator([train_dataset[i] for i in pilot_indices])
    pilot_batch = {k: v.to(model.device) for k, v in pilot_batch.items()}
    model.eval()
    with torch.no_grad():
        pilot_out = model(**pilot_batch)
    logger.info(
        "DIAGNOSTIC pre-flight: loss=%.4f, logits_mean=%.4f, "
        "logits_std=%.4f, any_nan=%s, seq_len=%d",
        pilot_out.loss.item(),
        pilot_out.logits.mean().item(),
        pilot_out.logits.std().item(),
        bool(torch.isnan(pilot_out.logits).any()),
        pilot_batch["input_ids"].shape[1],
    )
    if torch.isnan(pilot_out.logits).any():
        raise RuntimeError(
            f"NaN logits in pre-flight (seq_len={pilot_batch['input_ids'].shape[1]})! "
            "Model has numerical issues with this data."
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
