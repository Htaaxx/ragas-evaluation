"""
Generator training module for fine-tuning seq2seq models.

This module handles training the generator model (e.g., T5, FLAN-T5)
for question answering with retrieved contexts. Supports crash-safe
sub-epoch checkpointing with auto-resume.
"""

import math
import random
from pathlib import Path
from typing import List, Optional, Callable

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

from ..config import RAGConfig
from ..data.loader import TrainExample


class _GeneratorDataset(Dataset):
    """Dataset wrapper for generator training examples."""

    def __init__(self, examples: List[TrainExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainExample:
        return self.examples[idx]


class GeneratorTrainer:
    """
    Trainer for fine-tuning seq2seq generator models.

    The generator is trained to produce answers given:
    - Question
    - Retrieved contexts (from retriever)

    Training uses teacher forcing with cross-entropy loss.

    Supports crash-safe checkpointing: every ``checkpoint_steps`` steps the
    full training state (model weights, tokenizer, optimizer, scheduler, RNG
    states) is written to disk.  Re-running ``train()`` with
    ``resume_from_checkpoint=True`` (the default) picks up exactly where the
    previous run left off — already-processed batches are fast-skipped by
    restoring the epoch's RNG state.
    """

    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        config: RAGConfig,
        retrieval_fn: Optional[Callable[[str, int], List[str]]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the generator trainer.

        Args:
            model: Seq2seq model to train
            tokenizer: Tokenizer for the model
            config: Configuration object
            retrieval_fn: Function to retrieve contexts (question, top_k) -> contexts
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.retrieval_fn = retrieval_fn
        self.device = device or config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        print(f"Generator trainer initialized on {self.device}")

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    def create_dataloader(
        self,
        examples: List[TrainExample],
        batch_size: Optional[int] = None,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create DataLoader from training examples.

        Args:
            examples: List of TrainExample objects
            batch_size: Batch size (default: from config)
            shuffle: Whether to shuffle data

        Returns:
            DataLoader for training
        """
        batch_size = batch_size or self.config.generator_batch_size

        dataset = _GeneratorDataset(examples)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,           # Required on Windows; avoids multiprocessing crash
            collate_fn=lambda x: x,
        )

        print(f"Created DataLoader: {len(examples)} examples, batch_size={batch_size}")
        return dataloader

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        """
        Build input prompt from question and contexts.

        Args:
            question: The question
            contexts: List of context passages

        Returns:
            Formatted prompt string
        """
        context_str = "\n".join(f"- {ctx}" for ctx in contexts)
        return self.config.qa_prompt_template.format(
            context=context_str,
            question=question,
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
        optimizer,
        scheduler,
        epoch_losses: List[float],
        rng_states: dict,
    ) -> None:
        """
        Persist full training state so training can be resumed after a crash.

        The model and tokenizer are saved to a sub-folder (``generator_ckpt_model``).
        All scalar training state goes into ``generator_checkpoint.pt`` alongside it.

        Args:
            checkpoint_dir: Directory to write checkpoint into
            epoch: Current epoch index (0-based)
            step: Next step to execute when resuming (0-based)
            global_step: Total optimiser steps taken so far
            optimizer: AdamW optimizer
            scheduler: LR scheduler
            epoch_losses: Loss values accumulated in the current epoch so far
            rng_states: Dict with keys ``python``, ``numpy``, ``torch``, and
                        optionally ``cuda`` — captured at epoch start
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model + tokenizer
        model_ckpt_path = checkpoint_dir / "generator_ckpt_model"
        self.model.save_pretrained(model_ckpt_path)
        self.tokenizer.save_pretrained(model_ckpt_path)

        # Save scalar training state
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
        print(f"   [ckpt] Saved generator checkpoint → epoch={epoch+1}, step={step}")

    def load_checkpoint(
        self,
        checkpoint_dir: Path,
        optimizer,
        scheduler,
    ):
        """
        Load training state from a previously saved checkpoint.

        Restores optimizer, scheduler states in-place and reloads the model
        and tokenizer from the companion model folder.

        Args:
            checkpoint_dir: Directory containing the checkpoint
            optimizer: Optimizer to restore state into
            scheduler: LR scheduler to restore state into

        Returns:
            Tuple ``(start_epoch, start_step, global_step, epoch_losses,
            rng_states)`` or ``None`` if no checkpoint is found.
        """
        checkpoint_dir = Path(checkpoint_dir)
        state_path = checkpoint_dir / "generator_checkpoint.pt"
        model_ckpt_path = checkpoint_dir / "generator_ckpt_model"

        if not state_path.exists() or not model_ckpt_path.exists():
            return None

        print(f"Resuming generator from checkpoint: {state_path}")
        # weights_only=False required because the checkpoint contains numpy RNG
        # states (numpy._core.multiarray.scalar) which PyTorch 2.6+ blocks by
        # default. These files are written by us so they are trusted.
        state = torch.load(state_path, map_location=self.device, weights_only=False)

        # Reload model + tokenizer weights
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt_path)

        # Windows fix: safetensors files stay memory-mapped after loading,
        # which blocks overwriting them later (os error 1224). Cloning all
        # parameters moves data into regular RAM and drops the mmap references.
        # gc.collect() ensures Python immediately closes the file handles rather
        # than waiting for the next GC cycle.
        import gc
        with torch.no_grad():
            for param in self.model.parameters():
                param.data = param.data.clone()
            for buf in self.model.buffers():
                if buf is not None:
                    buf.data = buf.data.clone()
        gc.collect()

        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])

        print(
            f"   Resumed at epoch={state['epoch']+1}, step={state['step']}, "
            f"global_step={state['global_step']}"
        )
        return (
            state["epoch"],
            state["step"],
            state["global_step"],
            state["epoch_losses"],
            state["rng_states"],
        )

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
        """
        Train the generator model with crash-safe sub-epoch checkpointing.

        Every ``checkpoint_steps`` steps within an epoch the full training
        state is written to ``checkpoint_dir``. If ``resume_from_checkpoint``
        is ``True`` and a checkpoint is found there, training continues from
        the last saved step — already-processed batches are fast-skipped by
        restoring the epoch's RNG state and iterating without gradients.

        Args:
            train_loader: DataLoader with training examples
            epochs: Number of training epochs
            lr: Learning rate
            warmup_ratio: Warmup ratio for learning rate scheduler
            gradient_accumulation: Gradient accumulation steps
            max_input_tokens: Maximum input sequence length
            max_target_tokens: Maximum target sequence length
            top_k: Number of contexts to retrieve
            save_path: Path to save trained model (default: from config)
            resume_from_checkpoint: Auto-resume from last checkpoint if found
            checkpoint_dir: Where to write checkpoints (default: from config)
            checkpoint_steps: How often (in steps) to save a checkpoint
                              (default: from config)

        Returns:
            Trained model
        """
        # Resolve hyperparameters
        epochs = epochs or self.config.generator_epochs
        lr = lr or self.config.generator_lr
        warmup_ratio = warmup_ratio or self.config.generator_warmup_ratio
        gradient_accumulation = gradient_accumulation or self.config.generator_gradient_accumulation
        max_input_tokens = max_input_tokens or self.config.generator_max_input_tokens
        max_target_tokens = max_target_tokens or self.config.generator_max_target_tokens
        top_k = top_k or self.config.top_k
        save_path = save_path or self.config.get_generator_model_path()
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.config.get_checkpoint_dir()
        checkpoint_steps = checkpoint_steps or self.config.generator_checkpoint_steps

        # Enable gradient checkpointing to trade compute for memory
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # Build optimizer and scheduler based on total expected steps
        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = math.ceil(len(train_loader) * epochs / gradient_accumulation)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )

        start_epoch = 0
        start_step = 0
        global_step = 0
        saved_epoch_losses: List[float] = []
        saved_rng_states: Optional[dict] = None

        # ------------------------------------------------------------------
        # Auto-resume
        # ------------------------------------------------------------------
        if resume_from_checkpoint:
            result = self.load_checkpoint(checkpoint_dir, optimizer, scheduler)
            if result is not None:
                (
                    start_epoch,
                    start_step,
                    global_step,
                    saved_epoch_losses,
                    saved_rng_states,
                ) = result

        print(f"\nStarting generator training...")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {lr}")
        print(f"   Warmup ratio: {warmup_ratio}")
        print(f"   Gradient accumulation: {gradient_accumulation}")
        print(f"   Max input tokens: {max_input_tokens}")
        print(f"   Max target tokens: {max_target_tokens}")
        print(f"   Gradient checkpointing: enabled")
        print(f"   Checkpoint every {checkpoint_steps} steps → {checkpoint_dir}")

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        self.model.train()

        for epoch in range(start_epoch, epochs):
            # Carry over losses from a partially completed epoch on resume;
            # otherwise start fresh.
            epoch_losses: List[float] = list(saved_epoch_losses) if epoch == start_epoch else []

            # ----------------------------------------------------------
            # RNG state management — same principle as retriever trainer.
            # ----------------------------------------------------------
            if epoch == start_epoch and saved_rng_states is not None:
                random.setstate(saved_rng_states["python"])
                np.random.set_state(saved_rng_states["numpy"])
                torch.set_rng_state(saved_rng_states["torch"])
                if saved_rng_states.get("cuda") and torch.cuda.is_available():
                    torch.cuda.set_rng_state(saved_rng_states["cuda"])
                epoch_rng_states = saved_rng_states
            else:
                epoch_rng_states = {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                }
                start_step = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch_idx, batch in enumerate(progress):

                # ----------------------------------------------------------
                # Fast-skip already-processed batches (no gradient pass).
                # ----------------------------------------------------------
                if batch_idx < start_step:
                    progress.set_postfix({"status": f"skipping to step {start_step}"})
                    continue

                # Prepare prompts and targets
                batch_inputs = []
                batch_targets = []

                for example in batch:
                    if example.contexts:
                        contexts = list(example.contexts)
                    elif self.retrieval_fn:
                        contexts = self.retrieval_fn(example.question, top_k)
                    else:
                        raise ValueError(
                            "Either pre-compute contexts or provide retrieval_fn"
                        )
                    batch_inputs.append(self.build_prompt(example.question, contexts))
                    batch_targets.append(example.answer)

                # Tokenize inputs
                inputs = self.tokenizer(
                    batch_inputs,
                    padding=True,
                    truncation=True,
                    max_length=max_input_tokens,
                    return_tensors="pt",
                ).to(self.device)

                # Tokenize targets
                labels = self.tokenizer(
                    batch_targets,
                    padding=True,
                    truncation=True,
                    max_length=max_target_tokens,
                    return_tensors="pt",
                ).input_ids.to(self.device)

                # Ignore padding in loss
                labels[labels == self.tokenizer.pad_token_id] = -100

                # Forward + backward
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss / gradient_accumulation
                loss.backward()

                # Gradient accumulation step
                if (global_step + 1) % gradient_accumulation == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                raw_loss = loss.item() * gradient_accumulation
                epoch_losses.append(raw_loss)
                progress.set_postfix({"loss": f"{raw_loss:.4f}"})
                global_step += 1

                # ----------------------------------------------------------
                # Sub-epoch checkpoint
                # ----------------------------------------------------------
                if (batch_idx + 1) % checkpoint_steps == 0:
                    self.save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        epoch=epoch,
                        step=batch_idx + 1,   # next step to run on resume
                        global_step=global_step,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch_losses=epoch_losses,
                        rng_states=epoch_rng_states,
                    )

            # Clear intra-epoch resume state after epoch completes
            start_step = 0
            saved_epoch_losses = []
            saved_rng_states = None

            # Epoch statistics
            if epoch_losses:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                print(f"Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")

            # Epoch-end checkpoint (step=0 signals epoch boundary)
            self.save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch + 1,   # next epoch to run
                step=0,
                global_step=global_step,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch_losses=[],
                rng_states={
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                },
            )

        # Save final model
        self.model.eval()
        self.save_model(save_path)

        print(f"\nTraining complete!")
        print(f"Model saved to: {save_path}")
        return self.model

    # ------------------------------------------------------------------
    # Persistence helpers (used externally)
    # ------------------------------------------------------------------

    def save_model(self, save_path: Path) -> None:
        """
        Save the trained model and tokenizer.

        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")

    def load_model(self, load_path: Path) -> AutoModelForSeq2SeqLM:
        """
        Load a trained model and tokenizer.

        Args:
            load_path: Path to load the model from

        Returns:
            Loaded model
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(load_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        print(f"Model and tokenizer loaded from {load_path}")
        return self.model
