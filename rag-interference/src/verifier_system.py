"""
Self-RAG-style generative answer verifier.

Fine-tunes a seq2seq LM (Flan-T5) to generate reflection tokens
(IsRel, IsSup, IsUse) plus an ACCEPT/REJECT decision, given
(question, gold_context, candidate_answer) as input.

No retriever, no knowledge base — gold context is fed directly.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.filtering.data_split import load_and_split
from src.filtering.learned_filter import _extract_top1_context

from configs import load_config, resolve_repo_path

logger = logging.getLogger(__name__)

_TRUNCATION_STRATEGY = "only_first"


class _VerifierDataset(Dataset):
    """Seq2seq dataset for Self-RAG-style answer verification."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        cfg: Dict[str, Any],
        build_prompt,
        build_target,
        extract_context,
    ) -> None:
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.build_prompt = build_prompt
        self.build_target = build_target
        self.extract_context = extract_context
        self.rows = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows.iloc[idx]
        context = self.extract_context(str(row["context"]))
        prompt = self.build_prompt(
            str(row["question"]),
            context,
            str(row["answer"]),
        )
        target = self.build_target(int(row["label"]))

        model_cfg = self.cfg["model"]
        source = self.tokenizer(
            prompt,
            truncation=_TRUNCATION_STRATEGY,
            max_length=model_cfg["max_input_length"],
        )
        target_enc = self.tokenizer(
            text_target=target,
            truncation=True,
            max_length=model_cfg["max_target_length"],
        )
        source["labels"] = target_enc["input_ids"]
        return source


class VerifierSystem:
    """
    Self-RAG-style answer verifier — no retriever, no KB.

    Fine-tunes a seq2seq LM (Flan-T5) to generate reflection tokens
    (IsRel, IsSup, IsUse) plus an ACCEPT/REJECT decision, given
    (question, gold_context, candidate_answer) as input.

    Pipeline: load_data --> train --> load_model --> evaluate --> predict
    """

    def __init__(self, config_name: str = "rag_verifier") -> None:
        self.cfg = load_config(config_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        logger.info("VerifierSystem initialized on %s", self.device)

    def load_data(self, csv_path: Optional[str] = None) -> None:
        """Load labeled_asqa.csv and split into train / val / test."""
        data_cfg = self.cfg["data"]
        path = csv_path or str(resolve_repo_path(data_cfg["labeled_csv"]))
        self.train_df, self.val_df, self.test_df = load_and_split(
            csv_path=path,
            test_ratio=data_cfg["test_ratio"],
            val_ratio=data_cfg["val_ratio"],
            seed=data_cfg["seed"],
        )
        logger.info(
            "Data loaded: train=%d val=%d test=%d",
            len(self.train_df),
            len(self.val_df),
            len(self.test_df),
        )

    def _build_prompt(self, question: str, context: str, answer: str) -> str:
        return self.cfg["prompt_template"].format(
            question=question,
            context=context,
            answer=answer,
        )

    def _build_target(self, label: int) -> str:
        tokens = self.cfg["reflection_tokens"]
        if label == 1:
            return tokens["accept_target"]
        return tokens["reject_target"]

    def _extract_context(self, context_str: str) -> str:
        max_chars = self.cfg["data"]["context_max_chars"]
        return _extract_top1_context(context_str, max_chars=max_chars)

    def _make_dataset(self, df: pd.DataFrame) -> _VerifierDataset:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call train() or load_model() first.")
        return _VerifierDataset(
            df=df,
            tokenizer=self.tokenizer,
            cfg=self.cfg,
            build_prompt=self._build_prompt,
            build_target=self._build_target,
            extract_context=self._extract_context,
        )

    def train(self, resume_from_checkpoint: bool = False) -> Path:
        """Fine-tune Flan-T5 with Seq2SeqTrainer."""
        if self.train_df is None or self.val_df is None:
            raise RuntimeError("Call load_data() before train().")

        model_cfg = self.cfg["model"]
        train_cfg = self.cfg["training"]
        save_dir = resolve_repo_path(model_cfg["model_save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg["name"])

        train_dataset = self._make_dataset(self.train_df)
        val_dataset = self._make_dataset(self.val_df)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )

        args = Seq2SeqTrainingArguments(
            output_dir=str(save_dir),
            num_train_epochs=train_cfg["num_epochs"],
            per_device_train_batch_size=train_cfg["batch_size"],
            per_device_eval_batch_size=train_cfg["batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            warmup_ratio=train_cfg["warmup_ratio"],
            weight_decay=train_cfg["weight_decay"],
            max_grad_norm=train_cfg["max_grad_norm"],
            predict_with_generate=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=train_cfg["save_total_limit"],
            logging_steps=50,
            seed=train_cfg["seed"],
            fp16=train_cfg["fp16"],
            report_to="none",
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=train_cfg["early_stopping_patience"],
                ),
            ],
        )

        logger.info("Starting verifier training …")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        self.model = trainer.model
        logger.info("Verifier saved to %s", save_dir)
        return save_dir

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load a saved Flan-T5 verifier checkpoint."""
        model_cfg = self.cfg["model"]
        path = resolve_repo_path(model_path or model_cfg["model_save_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(path)).to(self.device)
        self.model.eval()
        logger.info("Verifier loaded from %s", path)

    def _generate_outputs(
        self,
        prompts: List[str],
        batch_size: int,
    ) -> List[str]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call train() or load_model() first.")

        outputs: List[str] = []
        max_input = self.cfg["model"]["max_input_length"]
        max_target = self.cfg["model"]["max_target_length"]

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            enc = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=_TRUNCATION_STRATEGY,
                max_length=max_input,
            ).to(self.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=max_target,
                    num_beams=1,
                )

            decoded = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
            outputs.extend(decoded)

        return outputs

    def predict(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> Dict[str, Any]:
        """Generate a Self-RAG reflection string and parse the decision."""
        context_text = self._extract_context(context)
        prompt = self._build_prompt(question, context_text, answer)
        raw_output = self._generate_outputs([prompt], batch_size=1)[0]
        parsed = self._parse_reflection(raw_output)
        parsed["raw_output"] = raw_output
        return parsed

    def predict_batch(
        self,
        questions: List[str],
        contexts: List[str],
        answers: List[str],
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        """Batch version of ``predict()``."""
        if not (len(questions) == len(contexts) == len(answers)):
            raise ValueError("questions, contexts, and answers must have equal length")

        prompts = [
            self._build_prompt(q, self._extract_context(c), a)
            for q, c, a in zip(questions, contexts, answers)
        ]
        raw_outputs = self._generate_outputs(prompts, batch_size=batch_size)
        results: List[Dict[str, Any]] = []
        for raw in raw_outputs:
            parsed = self._parse_reflection(raw)
            parsed["raw_output"] = raw
            results.append(parsed)
        return results

    def _parse_reflection(self, generated_text: str) -> Dict[str, Any]:
        """Parse generated reflection tokens; default to REJECT on failure."""
        tokens = self.cfg["reflection_tokens"]
        text = generated_text.strip()

        def _extract(tag: str) -> Optional[str]:
            pattern = rf"{re.escape(tag)}\s+(\S+)"
            match = re.search(pattern, text)
            return match.group(1) if match else None

        is_rel = _extract("[IsRel]")
        is_sup = _extract("[IsSup]")
        is_use = _extract("[IsUse]")

        decision = tokens["reject_keyword"]
        decision_match = re.search(
            rf"{re.escape(tokens['decision_keyword'])}\s+(\S+)",
            text,
        )
        if decision_match:
            decision = decision_match.group(1).upper()
        elif tokens["accept_keyword"] in text.upper():
            decision = tokens["accept_keyword"]
        elif tokens["reject_keyword"] in text.upper():
            decision = tokens["reject_keyword"]

        accept = decision == tokens["accept_keyword"]
        return {
            "decision": tokens["accept_keyword"] if accept else tokens["reject_keyword"],
            "accept": accept,
            "is_rel": is_rel,
            "is_sup": is_sup,
            "is_use": is_use,
        }

    def evaluate(self, split: str = "test") -> Dict[str, Any]:
        """Evaluate verifier on train / val / test split."""
        split_map = {
            "train": self.train_df,
            "val": self.val_df,
            "test": self.test_df,
        }
        df = split_map.get(split)
        if df is None:
            raise ValueError(f"Unknown split '{split}'. Use train, val, or test.")
        if df.empty:
            raise RuntimeError(f"Split '{split}' is empty. Call load_data() first.")
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load_model() first.")

        predictions = self.predict_batch(
            questions=df["question"].astype(str).tolist(),
            contexts=df["context"].astype(str).tolist(),
            answers=df["answer"].astype(str).tolist(),
        )

        y_true = df["label"].astype(int).tolist()
        y_pred = [1 if p["accept"] else 0 for p in predictions]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (
            int(cm[0, 0]),
            int(cm[0, 1]),
            int(cm[1, 0]),
            int(cm[1, 1]),
        )
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        expected_rel = "relevant"
        expected_sup = {1: "fully_supported", 0: "no_support"}
        expected_use = {1: "5", 0: "1"}

        rel_correct = sum(
            1 for p, lbl in zip(predictions, y_true)
            if p.get("is_rel") == expected_rel
        )
        sup_correct = sum(
            1 for p, lbl in zip(predictions, y_true)
            if p.get("is_sup") == expected_sup[lbl]
        )
        use_correct = sum(
            1 for p, lbl in zip(predictions, y_true)
            if p.get("is_use") == expected_use[lbl]
        )
        n_samples = len(y_true)

        metrics: Dict[str, Any] = {
            "split": split,
            "n_samples": n_samples,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "fpr": float(fpr),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "is_rel_accuracy": rel_correct / n_samples if n_samples else 0.0,
            "is_sup_accuracy": sup_correct / n_samples if n_samples else 0.0,
            "is_use_accuracy": use_correct / n_samples if n_samples else 0.0,
            "config_snapshot": self.cfg,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        results_dir = resolve_repo_path(self.cfg["evaluation"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"metrics_{split}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=str)

        logger.info(
            "Evaluation [%s]: acc=%.3f f1=%.3f fpr=%.3f "
            "IsRel=%.3f IsSup=%.3f IsUse=%.3f",
            split,
            metrics["accuracy"],
            metrics["f1"],
            metrics["fpr"],
            metrics["is_rel_accuracy"],
            metrics["is_sup_accuracy"],
            metrics["is_use_accuracy"],
        )
        logger.info("Metrics saved to %s", out_path)
        return metrics
