"""
Generator training module for fine-tuning seq2seq models.

Handles training the generator model (e.g., T5, FLAN-T5) for question
answering with retrieved contexts. Supports crash-safe sub-epoch
checkpointing with auto-resume.
"""

from __future__ import annotations

import gc
import logging
import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW

from rag_filtering.rag.config import RAGConfig
from rag_filtering.data.base_loader import TrainExample

logger = logging.getLogger(__name__)


class _GeneratorDataset(Dataset):
    """Dataset wrapper for generator training examples."""

    def __init__(self, examples: List[TrainExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainExample:
        return self.examples[idx]


class GeneratorTrainer:
    """
    Trainer for fine-tuning seq2seq generator models.

    Supports crash-safe checkpointing: every ``checkpoint_steps`` steps
    the full training state is written to disk. Re-running ``train()``
    with ``resume_from_checkpoint=True`` picks up exactly where the
    previous run left off.
    """

    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        config: RAGConfig,
        retrieval_fn: Optional[Callable[[str, int], List[str]]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.retrieval_fn = retrieval_fn
        self.device = (
            device or config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        logger.info("Generator trainer initialized on %s", self.device)

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    def create_dataloader(
        self,
        examples: List[TrainExample],
        batch_size: Optional[int] = None,
        shuffle: bool = True,
    ) -> DataLoader:
        batch_size = batch_size or self.config.generator_batch_size
        dataset = _GeneratorDataset(examples)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=lambda x: x,
        )
        logger.info(
            "Created DataLoader: %d examples, batch_size=%d",
            len(examples), batch_size,
        )
        return dataloader

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        context_str = "\n".join(f"- {ctx}" for ctx in contexts)
        return self.config.qa_prompt_template.format(
            context=context_str, question=question,
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        checkpoint_dir: Path,
        epoch: int,
        step: int,
        global_step: int,
        optimizer: AdamW,
        scheduler: object,
        epoch_losses: List[float],
        rng_states: dict,
    ) -> None:
        """Persist full training state for crash-safe resume."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_ckpt_path = checkpoint_dir / "generator_ckpt_model"
        self.model.save_pretrained(model_ckpt_path)
        self.tokenizer.save_pretrained(model_ckpt_path)

        state_path = checkpoint_dir / "generator_checkpoint.pt"
        torch.save(
            {
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch_losses": epoch_losses,
                "rng_states": rng_states,
            },
            state_path,
        )
        logger.info(
            "[ckpt] Saved generator checkpoint -> epoch=%d, step=%d",
            epoch + 1, step,
        )

    def load_checkpoint(
        self,
        checkpoint_dir: Path,
        optimizer: AdamW,
        scheduler: object,
    ) -> Optional[tuple]:
        """Load training state from a previously saved checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        state_path = checkpoint_dir / "generator_checkpoint.pt"
        model_ckpt_path = checkpoint_dir / "generator_ckpt_model"

        if not state_path.exists() or not model_ckpt_path.exists():
            return None

        logger.info("Resuming generator from checkpoint: %s", state_path)
        state = torch.load(
            state_path, map_location=self.device, weights_only=False
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_ckpt_path
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt_path)

        self._detach_mmap_handles()

        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])

        logger.info(
            "Resumed at epoch=%d, step=%d, global_step=%d",
            state["epoch"] + 1, state["step"], state["global_step"],
        )
        return (
            state["epoch"],
            state["step"],
            state["global_step"],
            state["epoch_losses"],
            state["rng_states"],
        )

    def _detach_mmap_handles(self) -> None:
        """Clone parameters to release safetensors mmap file handles (Windows)."""
        with torch.no_grad():
            for param in self.model.parameters():
                param.data = param.data.clone()
            for buf in self.model.buffers():
                if buf is not None:
                    buf.data = buf.data.clone()
        gc.collect()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        warmup_ratio: Optional[float] = None,
        gradient_accumulation: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
        max_target_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        save_path: Optional[Path] = None,
        resume_from_checkpoint: bool = True,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_steps: Optional[int] = None,
    ) -> AutoModelForSeq2SeqLM:
        """Train the generator model with crash-safe checkpointing."""
        hp = self._resolve_hyperparams(
            epochs, lr, warmup_ratio, gradient_accumulation,
            max_input_tokens, max_target_tokens, top_k,
            save_path, checkpoint_dir, checkpoint_steps,
        )

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        optimizer = AdamW(self.model.parameters(), lr=hp["lr"])
        total_steps = math.ceil(
            len(train_loader) * hp["epochs"] / hp["gradient_accumulation"]
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * hp["warmup_ratio"]),
            num_training_steps=total_steps,
        )

        start_epoch, start_step, global_step = 0, 0, 0
        saved_epoch_losses: List[float] = []
        saved_rng_states: Optional[dict] = None

        if resume_from_checkpoint:
            result = self.load_checkpoint(
                hp["checkpoint_dir"], optimizer, scheduler
            )
            if result is not None:
                (start_epoch, start_step, global_step,
                 saved_epoch_losses, saved_rng_states) = result

        self._log_training_config(hp)
        self.model.train()

        for epoch in range(start_epoch, hp["epochs"]):
            epoch_losses = (
                list(saved_epoch_losses) if epoch == start_epoch else []
            )

            epoch_rng = self._manage_rng(
                epoch, start_epoch, saved_rng_states
            )
            if epoch != start_epoch:
                start_step = 0

            global_step = self._run_epoch(
                train_loader, optimizer, scheduler, epoch, hp,
                start_step, global_step, epoch_losses, epoch_rng,
            )

            start_step = 0
            saved_epoch_losses = []
            saved_rng_states = None

            if epoch_losses:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                logger.info(
                    "Epoch %d/%d: avg_loss=%.4f",
                    epoch + 1, hp["epochs"], avg_loss,
                )

            self.save_checkpoint(
                checkpoint_dir=hp["checkpoint_dir"],
                epoch=epoch + 1,
                step=0,
                global_step=global_step,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch_losses=[],
                rng_states=self._capture_rng(),
            )

        self.model.eval()
        self.save_model(hp["save_path"])
        logger.info("Training complete! Model saved to: %s", hp["save_path"])
        return self.model

    # ------------------------------------------------------------------
    # Training internals
    # ------------------------------------------------------------------

    def _resolve_hyperparams(self, *args) -> Dict:
        (epochs, lr, warmup_ratio, gradient_accumulation,
         max_input_tokens, max_target_tokens, top_k,
         save_path, checkpoint_dir, checkpoint_steps) = args
        return {
            "epochs": epochs or self.config.generator_epochs,
            "lr": lr or self.config.generator_lr,
            "warmup_ratio": warmup_ratio or self.config.generator_warmup_ratio,
            "gradient_accumulation": (
                gradient_accumulation
                or self.config.generator_gradient_accumulation
            ),
            "max_input_tokens": (
                max_input_tokens or self.config.generator_max_input_tokens
            ),
            "max_target_tokens": (
                max_target_tokens or self.config.generator_max_target_tokens
            ),
            "top_k": top_k or self.config.top_k,
            "save_path": save_path or self.config.get_generator_model_path(),
            "checkpoint_dir": (
                Path(checkpoint_dir) if checkpoint_dir
                else self.config.get_checkpoint_dir()
            ),
            "checkpoint_steps": (
                checkpoint_steps or self.config.generator_checkpoint_steps
            ),
        }

    def _log_training_config(self, hp: Dict) -> None:
        logger.info("Starting generator training …")
        for k, v in hp.items():
            logger.info("   %s: %s", k, v)
        logger.info("   Gradient checkpointing: enabled")

    def _run_epoch(
        self,
        train_loader: DataLoader,
        optimizer: AdamW,
        scheduler: object,
        epoch: int,
        hp: Dict,
        start_step: int,
        global_step: int,
        epoch_losses: List[float],
        epoch_rng: dict,
    ) -> int:
        """Run one training epoch, returns updated global_step."""
        progress = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{hp['epochs']}"
        )

        for batch_idx, batch in enumerate(progress):
            if batch_idx < start_step:
                progress.set_postfix(
                    {"status": f"skipping to step {start_step}"}
                )
                continue

            batch_inputs, batch_targets = self._prepare_batch(batch, hp)

            inputs = self.tokenizer(
                batch_inputs,
                padding=True,
                truncation=True,
                max_length=hp["max_input_tokens"],
                return_tensors="pt",
            ).to(self.device)

            labels = self.tokenizer(
                batch_targets,
                padding=True,
                truncation=True,
                max_length=hp["max_target_tokens"],
                return_tensors="pt",
            ).input_ids.to(self.device)

            labels[labels == self.tokenizer.pad_token_id] = -100

            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss / hp["gradient_accumulation"]
            loss.backward()

            if (global_step + 1) % hp["gradient_accumulation"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            raw_loss = loss.item() * hp["gradient_accumulation"]
            epoch_losses.append(raw_loss)
            progress.set_postfix({"loss": f"{raw_loss:.4f}"})
            global_step += 1

            if (batch_idx + 1) % hp["checkpoint_steps"] == 0:
                self.save_checkpoint(
                    checkpoint_dir=hp["checkpoint_dir"],
                    epoch=epoch,
                    step=batch_idx + 1,
                    global_step=global_step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch_losses=epoch_losses,
                    rng_states=epoch_rng,
                )

        return global_step

    def _prepare_batch(
        self, batch: List[TrainExample], hp: Dict
    ) -> tuple[list[str], list[str]]:
        batch_inputs: List[str] = []
        batch_targets: List[str] = []
        for example in batch:
            if example.contexts:
                contexts = list(example.contexts)
            elif self.retrieval_fn:
                contexts = self.retrieval_fn(example.question, hp["top_k"])
            else:
                raise ValueError(
                    "Either pre-compute contexts or provide retrieval_fn"
                )
            batch_inputs.append(self.build_prompt(example.question, contexts))
            batch_targets.append(example.answer)
        return batch_inputs, batch_targets

    @staticmethod
    def _manage_rng(
        epoch: int, start_epoch: int, saved_rng: Optional[dict]
    ) -> dict:
        if epoch == start_epoch and saved_rng is not None:
            random.setstate(saved_rng["python"])
            np.random.set_state(saved_rng["numpy"])
            torch.set_rng_state(saved_rng["torch"])
            if saved_rng.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state(saved_rng["cuda"])
            return saved_rng
        return GeneratorTrainer._capture_rng()

    @staticmethod
    def _capture_rng() -> dict:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state()
                if torch.cuda.is_available()
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_model(self, save_path: Path) -> None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("Model and tokenizer saved to %s", save_path)

    def load_model(self, load_path: Path) -> AutoModelForSeq2SeqLM:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            load_path
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        logger.info("Model and tokenizer loaded from %s", load_path)
        return self.model
