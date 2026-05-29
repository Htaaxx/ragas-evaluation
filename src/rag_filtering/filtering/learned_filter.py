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
import math
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
    TrainerCallback,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from .data_models import FilterDecision

from rag_filtering.config.loader import FILTERING_CONFIG, load_config_section

logger = logging.getLogger(__name__)


def _load_learned_filter_config() -> dict:
    """Load the ``learned_filter`` section from the filtering config."""
    return load_config_section(FILTERING_CONFIG, "learned_filter")


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
# NaN diagnostics — pinpoint where/when training diverges
# ---------------------------------------------------------------------------

class _NaNDiagnosticCallback(TrainerCallback):
    """Logs per-step health and flags the FIRST non-finite weight/grad.

    Used to debug the overfit NaN explosion: it reports, at the exact step
    things break, whether the GRADIENTS or the WEIGHTS went non-finite first
    (gradients first => exploding loss / bad input; weights first with finite
    grads => optimizer update instability, e.g. AdamW eps too small once the
    gradient vanishes on an overfit set).
    """

    def __init__(self, verbose_until: int = 60, every: int = 20) -> None:
        # Log grad-norm every step up to ``verbose_until`` (catch early
        # divergence), then only every ``every`` steps to avoid flooding
        # multi-thousand-step full-training runs. Non-finite events are
        # ALWAYS logged regardless of throttling.
        self._verbose_until = verbose_until
        self._every = every
        self._weight_reported = False
        self._grad_reported = False

    def _scan(self, model, kind: str):
        """Return (name, stat) of the first non-finite tensor of ``kind``."""
        for name, p in model.named_parameters():
            tensor = p.grad if kind == "grad" else p.data
            if tensor is None:
                continue
            if not torch.isfinite(tensor).all():
                finite = tensor[torch.isfinite(tensor)]
                amax = float(finite.abs().max()) if finite.numel() else float("nan")
                return name, amax
        return None, None

    @staticmethod
    def _total_grad_norm(model) -> float:
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                if torch.isfinite(g).all():
                    total += float(g.norm(2)) ** 2
                else:
                    return float("inf")
        return total ** 0.5

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Grads are populated here (BEFORE clipping + the optimizer step)."""
        model = kwargs.get("model")
        if model is None:
            return
        gnorm = self._total_grad_norm(model)
        # Per-step grad-norm trace: rising -> divergence; ~0 -> vanished.
        step = state.global_step
        non_finite_norm = not math.isfinite(gnorm)
        if step <= self._verbose_until or step % self._every == 0 or non_finite_norm:
            logger.info(
                "NaN-DIAG step %d: raw_grad_norm(pre-clip)=%.4e",
                step, gnorm,
            )
        gname, gmax = self._scan(model, "grad")
        if gname is not None and not self._grad_reported:
            logger.error(
                "NaN-DIAG: NON-FINITE GRAD first at step %d in '%s' "
                "(max finite |grad|=%.3e) -> exploding loss / bad forward, "
                "NOT the optimizer update.",
                state.global_step, gname, gmax,
            )
            self._grad_reported = True

    def on_step_end(self, args, state, control, **kwargs):
        """Weights are post-update here."""
        model = kwargs.get("model")
        if model is None or self._weight_reported:
            return
        wname, wmax = self._scan(model, "weight")
        if wname is not None:
            logger.error(
                "NaN-DIAG: NON-FINITE WEIGHT first at step %d in '%s' "
                "(max finite |w|=%.3e). If grads were finite this step, the "
                "OPTIMIZER UPDATE blew up (suspect AdamW eps too small once "
                "the gradient vanished on the overfit set).",
                state.global_step, wname, wmax,
            )
            self._weight_reported = True


def _log_head_init_stats(model) -> None:
    """Log the freshly-initialized classifier head stats (sanity on reinit)."""
    for name, p in model.named_parameters():
        if name.startswith("classifier"):
            logger.info(
                "NaN-DIAG: init '%s' shape=%s mean=%.4e std=%.4e max|.|=%.4e "
                "finite=%s",
                name, tuple(p.shape), float(p.mean()), float(p.std()),
                float(p.abs().max()), bool(torch.isfinite(p).all()),
            )


# ---------------------------------------------------------------------------
# Overfit sanity check (Step 5) — the HARD GATE before full training
# ---------------------------------------------------------------------------

def overfit_sanity_check(
    df: pd.DataFrame,
    n_pairs: int = 16,
    epochs: int = 50,
    model_name: str | None = None,
    learning_rate: float | None = None,
    max_length: int | None = None,
    batch_size: int | None = None,
) -> Dict[str, float]:
    """Try to overfit ``n_pairs`` paired (correct, halluc) examples.

    A 184M-parameter model MUST be able to fit a tiny labelled set. If
    train F1 does not reach >= 0.95 within ``epochs``, the code, data,
    or input format is broken — do NOT scale up to full training.

    Uses the EXACT SAME training stack as ``train_classifier`` —
    HuggingFace ``Trainer`` + ``TrainingArguments`` +
    ``DataCollatorWithPadding`` + ``_QADataset`` — so the overfit gate
    is apples-to-apples with full training. If overfit passes, full
    training will use the same machinery and should also work. If
    overfit fails, the bug is in the model/data/format, not the
    training loop.

    Parameters
    ----------
    df:
        DataFrame with ``id``, ``question``, ``answer``, ``context``,
        ``label`` columns.
    n_pairs:
        Number of paired base IDs to use (total samples = 2 * n_pairs).
    epochs:
        Training epochs on the tiny set.
    model_name / learning_rate / max_length / batch_size:
        Default to ``filtering.yaml`` values so the overfit run
        mirrors the full-training hyperparameters exactly.

    Returns
    -------
    Dict with ``train_f1``, ``train_precision``, ``train_recall``,
    ``train_fpr``, ``n_samples``, and confusion-matrix counts
    (``tp/tn/fp/fn``). The notebook gate cell asserts
    ``train_f1 >= 0.95``; if it fails, the 5 worst-classified
    pairs are logged with their IDs and predicted probabilities.
    """
    cfg = _load_learned_filter_config()
    model_name = model_name or cfg["model_name"]
    max_length = max_length or cfg.get("max_length", 512)
    # NOTE: overfit tests INTENTIONALLY use a higher LR than full training.
    # full-training lr (cfg[learning_rate]=1e-5) is tuned to generalize over
    # thousands of samples; the overfit gate runs on 32 samples with a
    # freshly re-initialized 2-class head (the MNLI 3-class head is dropped),
    # which needs an aggressive LR to escape the trivial constant-output
    # local minimum (loss stuck at ln 2 ≈ 0.693). 5e-5 is the standard
    # overfit-test LR for DeBERTa-base.
    if learning_rate is None:
        learning_rate = 5e-5
    if batch_size is None:
        batch_size = cfg.get("batch_size", 4)

    # --- Paired-sample selection (same logic as before) ---
    base_col = df["id"].str.replace(r"b$", "", regex=True)
    df_idx = df.copy()
    df_idx["_base"] = base_col

    selected_rows = []
    seen_bases: set = set()
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
        "OVERFIT TEST: %d samples (%d pos, %d neg) for %d epochs "
        "at lr=%.0e, batch_size=%d, max_length=%d, using HF Trainer "
        "(same stack as train_classifier)",
        len(mini_df),
        sum(1 for lbl in labels if lbl == 1),
        sum(1 for lbl in labels if lbl == 0),
        epochs, learning_rate, batch_size, max_length,
    )

    # --- Model + tokenizer + dataset (same _QADataset as train_classifier) ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = _load_deberta_model(model_name, num_labels=2)

    # NaN-DIAG: confirm the reinitialized 2-class head is sane (finite, small).
    _log_head_init_stats(model)

    overfit_dataset = _QADataset(
        contexts, answers, labels, tokenizer, max_length=max_length,
    )
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True,
    )

    # --- TrainingArguments: mirror train_classifier defaults ---
    # Differences from full training:
    #  - save_strategy="no" (we don't need checkpoints for a sanity gate)
    #  - eval_strategy="no" (we do our own train-set eval below)
    #  - num_train_epochs=epochs (50, not 10)
    #  - gradient_accumulation_steps=1 (tiny dataset, no need to accumulate)
    tmp_out = Path("/tmp/overfit_check_ckpt")
    tmp_out.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(tmp_out),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=cfg.get("weight_decay", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        # NaN-DIAG: log EVERY step and do NOT hide nan/inf losses, so the raw
        # loss/grad_norm trajectory is visible up to the exact divergence step.
        logging_steps=1,
        logging_nan_inf_filter=False,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=cfg.get("seed", 42),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=overfit_dataset,
        data_collator=data_collator,
        callbacks=[_NaNDiagnosticCallback()],
    )

    # NaN-DIAG: inspect the first tokenized batch (input id range, seq len,
    # label balance) — a bad input range is a classic silent NaN source.
    _diag_batch = data_collator([overfit_dataset[i] for i in range(min(batch_size, len(overfit_dataset)))])
    logger.info(
        "NaN-DIAG: first batch input_ids[min=%d max=%d] seq_len=%d "
        "vocab_size=%d attn_sum=%s labels=%s",
        int(_diag_batch["input_ids"].min()),
        int(_diag_batch["input_ids"].max()),
        int(_diag_batch["input_ids"].shape[1]),
        int(getattr(model.config, "vocab_size", -1)),
        _diag_batch["attention_mask"].sum(dim=1).tolist()
        if "attention_mask" in _diag_batch else "n/a",
        _diag_batch.get("labels").tolist() if "labels" in _diag_batch else "n/a",
    )

    logger.info(
        "Starting overfit training via HF Trainer "
        "(NaN-DIAG: logging_steps=1, nan_inf_filter=OFF) ..."
    )
    train_result = trainer.train()
    logger.info(
        "Overfit training done: %d steps, final train loss=%.4f",
        int(train_result.global_step),
        float(train_result.training_loss),
    )

    # --- Final evaluation on the same 32 train samples ---
    # Use the trained model (now reloaded into the trainer) in eval mode.
    eval_model = trainer.model
    eval_model.eval()
    device = next(eval_model.parameters()).device

    all_preds: List[int] = []
    all_probs: List[float] = []
    with torch.no_grad():
        for i in range(len(overfit_dataset)):
            sample = data_collator([overfit_dataset[i]])
            # Drop labels for the forward pass (we have them in ``labels``)
            model_inputs = {
                k: v.to(device) for k, v in sample.items() if k != "labels"
            }
            out = eval_model(**model_inputs)
            probs = torch.softmax(out.logits, dim=-1)[0]
            all_probs.append(probs[1].item())
            all_preds.append(int(probs.argmax().item()))

    cm = confusion_matrix(labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = (
        int(cm[0, 0]), int(cm[0, 1]),
        int(cm[1, 0]), int(cm[1, 1]),
    )
    f1 = f1_score(labels, all_preds, zero_division=0)
    precision = precision_score(labels, all_preds, zero_division=0)
    recall = recall_score(labels, all_preds, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    logger.info(
        "OVERFIT RESULT: F1=%.3f P=%.3f R=%.3f FPR=%.3f "
        "[TN=%d FP=%d FN=%d TP=%d]",
        f1, precision, recall, fpr, tn, fp, fn, tp,
    )

    # If it failed, log the 5 worst-classified pairs for debugging
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


# ---------------------------------------------------------------------------
# Differential learning-rate optimizer (root-cause fix)
# ---------------------------------------------------------------------------

def _build_differential_optimizer(
    model,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    num_training_steps: int,
    warmup_ratio: float,
):
    """Build AdamW with differential LRs + a linear warmup scheduler.

    The MNLI 3-class head is dropped at load time and replaced by a
    randomly-initialized 2-class head. A pretrained backbone needs a small
    LR (``backbone_lr``) for stability, but the from-scratch head needs a
    larger LR (``head_lr``) to escape the trivial constant-output minimum
    (loss stuck at ln 2 ≈ 0.693). A single shared LR of 1e-5 collapsed the
    model to all-reject; separating the two LRs is the fix.

    LayerNorm weights and biases are excluded from weight decay (standard
    practice, also avoids penalizing the fresh head's bias). A matching
    ``get_linear_schedule_with_warmup`` scheduler is always returned with
    the optimizer — never pass an optimizer without its scheduler.
    """
    def _is_head(param_name: str) -> bool:
        return param_name.startswith("classifier")

    def _no_decay(param_name: str) -> bool:
        return "bias" in param_name or "LayerNorm" in param_name

    groups = {
        "backbone_decay": [],
        "backbone_nodecay": [],
        "head_decay": [],
        "head_nodecay": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        prefix = "head" if _is_head(name) else "backbone"
        suffix = "nodecay" if _no_decay(name) else "decay"
        groups[f"{prefix}_{suffix}"].append(param)

    optimizer_grouped_parameters = [
        {"params": groups["backbone_decay"], "lr": backbone_lr,
         "weight_decay": weight_decay},
        {"params": groups["backbone_nodecay"], "lr": backbone_lr,
         "weight_decay": 0.0},
        {"params": groups["head_decay"], "lr": head_lr,
         "weight_decay": weight_decay},
        {"params": groups["head_nodecay"], "lr": head_lr,
         "weight_decay": 0.0},
    ]

    # eps=1e-6 (not the 1e-8 default) for DeBERTa-v3 numerical stability.
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-6)

    warmup_steps = int(warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    logger.info(
        "Differential LR active: backbone=%.0e head=%.0e "
        "(warmup_steps=%d / total_steps=%d)",
        backbone_lr, head_lr, warmup_steps, num_training_steps,
    )
    return optimizer, scheduler


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
    classifier_lr: float = cfg.get("classifier_lr", learning_rate)
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

    # NaN-DIAG: confirm the reinitialized 2-class head is sane (finite, small).
    _log_head_init_stats(model)

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
        # NaN-DIAG: never mask nan/inf losses as 0.0 — show the real value.
        logging_nan_inf_filter=False,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        save_total_limit=save_total_limit,
        seed=seed,
        report_to="none",
    )

    # --- Differential-LR optimizer + scheduler (root-cause fix) ---
    # A single shared LR (1e-5) left the randomly-initialized 2-class head
    # undertrained and collapsed the model to all-reject. Give the backbone
    # the small LR and the fresh head a larger one, with a matching warmup
    # scheduler. Steps must match the TrainingArguments grad-accum setting.
    grad_accum = max(1, 16 // batch_size)
    steps_per_epoch = math.ceil(len(train_dataset) / (batch_size * grad_accum))
    num_training_steps = steps_per_epoch * num_epochs
    optimizer, scheduler = _build_differential_optimizer(
        model,
        backbone_lr=learning_rate,
        head_lr=classifier_lr,
        weight_decay=weight_decay,
        num_training_steps=num_training_steps,
        warmup_ratio=warmup_ratio,
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
            optimizers=(optimizer, scheduler),
            callbacks=[
                _NaNDiagnosticCallback(),
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
            optimizers=(optimizer, scheduler),
            callbacks=[
                _NaNDiagnosticCallback(),
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
        "logits_std=%.4f, any_nan=%s, seq_len=%d, "
        "input_ids[min=%d max=%d], vocab_size=%d",
        pilot_out.loss.item(),
        pilot_out.logits.mean().item(),
        pilot_out.logits.std().item(),
        bool(torch.isnan(pilot_out.logits).any()),
        pilot_batch["input_ids"].shape[1],
        int(pilot_batch["input_ids"].min()),
        int(pilot_batch["input_ids"].max()),
        int(getattr(model.config, "vocab_size", -1)),
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
