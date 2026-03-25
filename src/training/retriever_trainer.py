"""
Retriever training module for fine-tuning sentence transformers.

This module handles training the retriever model using contrastive learning
on question-passage pairs from HotpotQA / ASQA. Supports crash-safe
sub-epoch checkpointing with auto-resume.
"""

import itertools
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer
from torch import nn
from torch.optim import AdamW
try:
    from torch.amp import GradScaler, autocast  # PyTorch >= 2.4
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # PyTorch < 2.4
    _AMP_DEVICE = "cuda"
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import RAGConfig
from ..data.loader import RetrieverExample


class RetrieverTrainer:
    """
    Trainer for fine-tuning retriever models using contrastive learning.

    The training uses a contrastive loss where:
    - Positive pairs: (question, relevant_passage)
    - Negative pairs: (question, irrelevant_passage)

    The model learns to maximize similarity between positive pairs
    and minimize similarity between negative pairs.

    Supports crash-safe checkpointing: every ``checkpoint_steps`` steps the
    full training state (model weights, optimizer, scaler, RNG states) is
    written to disk. Re-running ``train()`` with ``resume_from_checkpoint=True``
    (the default) picks up exactly where the previous run left off.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        config: RAGConfig,
        device: Optional[str] = None
    ):
        """
        Initialize the retriever trainer.

        Args:
            model: SentenceTransformer model to train
            config: Configuration object
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.config = config
        self.device = device or config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = min(
                self.model.max_seq_length,
                config.retriever_max_seq_length
            )

        print(f"Retriever trainer initialized on {self.device}")

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    def create_dataloader(
        self,
        examples: List[RetrieverExample],
        batch_size: Optional[int] = None,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create DataLoader from retriever examples.

        Args:
            examples: List of RetrieverExample objects
            batch_size: Batch size (default: from config)
            shuffle: Whether to shuffle data

        Returns:
            DataLoader for training
        """
        batch_size = batch_size or self.config.retriever_batch_size

        input_examples = [
            InputExample(texts=[ex.question, ex.positive_passage])
            for ex in examples
        ]

        dataloader = DataLoader(
            input_examples,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=0,           # Required on Windows; avoids multiprocessing crash
            collate_fn=lambda x: x,
        )

        print(f"Created DataLoader: {len(input_examples)} examples, batch_size={batch_size}")
        return dataloader

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        checkpoint_dir: Path,
        epoch: int,
        step: int,
        optimizer: AdamW,
        scaler: GradScaler,
        best_loss: float,
        patience_counter: int,
        rng_states: dict,
    ) -> None:
        """
        Persist full training state so training can be resumed after a crash.

        The SentenceTransformer model is saved to a sub-folder (it does not
        expose a standard ``state_dict``). Everything else goes into a single
        ``.pt`` file alongside it.

        Args:
            checkpoint_dir: Directory to write checkpoint into
            epoch: Current epoch index (0-based)
            step: Current step within the epoch (0-based, *exclusive* — the
                  next step to run when resuming)
            optimizer: AdamW optimizer
            scaler: GradScaler for mixed-precision training
            best_loss: Best average loss seen so far (for early stopping)
            patience_counter: Early-stopping patience counter
            rng_states: Dict with keys ``python``, ``numpy``, ``torch``, and
                        optionally ``cuda`` — captured at epoch start so we
                        can replay the same shuffle on resume
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save SentenceTransformer weights into a sub-folder
        model_ckpt_path = checkpoint_dir / "retriever_ckpt_model"
        self.model.save(str(model_ckpt_path))

        # Save scalar training state
        state_path = checkpoint_dir / "retriever_checkpoint.pt"
        torch.save(
            {
                "epoch": epoch,
                "step": step,
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_loss": best_loss,
                "patience_counter": patience_counter,
                "rng_states": rng_states,
            },
            state_path,
        )
        print(f"   [ckpt] Saved retriever checkpoint → epoch={epoch+1}, step={step}")

    def load_checkpoint(
        self,
        checkpoint_dir: Path,
        optimizer: AdamW,
        scaler: GradScaler,
    ):
        """
        Load training state from a previously saved checkpoint.

        Restores optimizer and scaler states in-place and reloads the model
        weights from the companion model folder.

        Args:
            checkpoint_dir: Directory containing the checkpoint
            optimizer: Optimizer to restore state into
            scaler: GradScaler to restore state into

        Returns:
            Tuple ``(start_epoch, start_step, best_loss, patience_counter,
            rng_states)`` or ``None`` if no checkpoint is found.
        """
        checkpoint_dir = Path(checkpoint_dir)
        state_path = checkpoint_dir / "retriever_checkpoint.pt"
        model_ckpt_path = checkpoint_dir / "retriever_ckpt_model"

        if not state_path.exists() or not model_ckpt_path.exists():
            return None

        print(f"Resuming retriever from checkpoint: {state_path}")
        # weights_only=False required because the checkpoint contains numpy RNG
        # states (numpy._core.multiarray.scalar) which PyTorch 2.6+ blocks by
        # default. These files are written by us so they are trusted.
        state = torch.load(state_path, map_location=self.device, weights_only=False)

        # Reload model weights
        self.model = SentenceTransformer(str(model_ckpt_path), device=self.device)
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = min(
                self.model.max_seq_length,
                self.config.retriever_max_seq_length,
            )

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
        scaler.load_state_dict(state["scaler_state"])

        print(
            f"   Resumed at epoch={state['epoch']+1}, step={state['step']}, "
            f"best_loss={state['best_loss']:.4f}"
        )
        return (
            state["epoch"],
            state["step"],
            state["best_loss"],
            state["patience_counter"],
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
        patience: Optional[int] = None,
        temperature: Optional[float] = None,
        accumulation_steps: Optional[int] = None,
        use_fp16: Optional[bool] = None,
        save_path: Optional[Path] = None,
        resume_from_checkpoint: bool = True,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_steps: Optional[int] = None,
    ) -> SentenceTransformer:
        """
        Train the retriever model with crash-safe sub-epoch checkpointing.

        Every ``checkpoint_steps`` steps within an epoch the full training
        state is written to ``checkpoint_dir``. If ``resume_from_checkpoint``
        is ``True`` and a checkpoint is found there, training continues from
        the last saved step — the already-processed batches are fast-skipped
        by restoring the epoch's RNG state and iterating without gradients.

        Args:
            train_loader: DataLoader with training examples
            epochs: Number of training epochs
            lr: Learning rate
            patience: Early stopping patience
            temperature: Temperature scaling for similarity scores
            accumulation_steps: Gradient accumulation steps
            use_fp16: Whether to use mixed precision training
            save_path: Path to save best model (default: from config)
            resume_from_checkpoint: Auto-resume from last checkpoint if found
            checkpoint_dir: Where to write checkpoints (default: from config)
            checkpoint_steps: How often (in steps) to save a checkpoint
                              (default: from config)

        Returns:
            Trained SentenceTransformer model
        """
        # Resolve hyperparameters
        epochs = epochs or self.config.retriever_epochs
        lr = lr or self.config.retriever_lr
        patience = patience or self.config.retriever_patience
        temperature = temperature or self.config.retriever_temperature
        accumulation_steps = accumulation_steps or self.config.retriever_accumulation_steps
        use_fp16 = use_fp16 if use_fp16 is not None else self.config.retriever_use_fp16
        save_path = save_path or self.config.get_retriever_model_path()
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.config.get_checkpoint_dir()
        checkpoint_steps = checkpoint_steps or self.config.retriever_checkpoint_steps

        # Setup optimizer / loss / scaler
        optimizer = AdamW(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        scaler = GradScaler(_AMP_DEVICE, enabled=use_fp16)

        best_loss = float("inf")
        patience_counter = 0
        start_epoch = 0
        start_step = 0
        saved_rng_states: Optional[dict] = None

        # ------------------------------------------------------------------
        # Auto-resume
        # ------------------------------------------------------------------
        if resume_from_checkpoint:
            result = self.load_checkpoint(checkpoint_dir, optimizer, scaler)
            if result is not None:
                start_epoch, start_step, best_loss, patience_counter, saved_rng_states = result

        print(f"\nStarting retriever training...")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {lr}")
        print(f"   Temperature: {temperature}")
        print(f"   Accumulation steps: {accumulation_steps}")
        print(f"   Mixed precision: {use_fp16}")
        print(f"   Checkpoint every {checkpoint_steps} steps → {checkpoint_dir}")

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        for epoch in range(start_epoch, epochs):
            epoch_losses = []

            # ----------------------------------------------------------
            # Capture / restore RNG state so the DataLoader shuffle is
            # reproducible across runs (required for correct batch-skip).
            # ----------------------------------------------------------
            if epoch == start_epoch and saved_rng_states is not None:
                # Restore the RNG state from the checkpoint so we get the
                # same shuffle order as when the checkpoint was created.
                random.setstate(saved_rng_states["python"])
                np.random.set_state(saved_rng_states["numpy"])
                torch.set_rng_state(saved_rng_states["torch"])
                if saved_rng_states.get("cuda") and torch.cuda.is_available():
                    torch.cuda.set_rng_state(saved_rng_states["cuda"])
                epoch_rng_states = saved_rng_states
            else:
                # Fresh epoch — capture current RNG state before DataLoader
                # starts shuffling so we can replay it later if needed.
                epoch_rng_states = {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                }
                start_step = 0  # Reset intra-epoch offset for fresh epochs

            self.model.train()
            optimizer.zero_grad(set_to_none=True)

            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for step, batch in enumerate(progress):

                # ----------------------------------------------------------
                # Fast-skip batches already processed before the crash.
                # We iterate through them without computing gradients so
                # that the DataLoader's internal shuffle state stays in sync.
                # ----------------------------------------------------------
                if step < start_step:
                    progress.set_postfix({"status": f"skipping to step {start_step}"})
                    continue

                # Extract questions and passages
                questions = [ex.texts[0] for ex in batch]
                passages = [ex.texts[1] for ex in batch]

                # Tokenize
                q_features = self.model.tokenize(questions)
                p_features = self.model.tokenize(passages)

                q_features = {k: v.to(self.device) for k, v in q_features.items()}
                p_features = {k: v.to(self.device) for k, v in p_features.items()}

                # Forward pass
                with autocast(_AMP_DEVICE, enabled=use_fp16):
                    q_emb = self.model(q_features)["sentence_embedding"]
                    p_emb = self.model(p_features)["sentence_embedding"]

                    sim = torch.matmul(q_emb, p_emb.T) * temperature
                    labels = torch.arange(sim.size(0)).to(self.device)
                    loss = loss_fn(sim, labels) / accumulation_steps

                # Backward pass
                scaler.scale(loss).backward()

                # Gradient accumulation
                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                epoch_losses.append(loss.item() * accumulation_steps)
                progress.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

                # ----------------------------------------------------------
                # Sub-epoch checkpoint
                # ----------------------------------------------------------
                if (step + 1) % checkpoint_steps == 0:
                    self.save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        epoch=epoch,
                        step=step + 1,   # "next step to run" on resume
                        optimizer=optimizer,
                        scaler=scaler,
                        best_loss=best_loss,
                        patience_counter=patience_counter,
                        rng_states=epoch_rng_states,
                    )

            # After the epoch finishes the intra-epoch offset is cleared
            start_step = 0
            saved_rng_states = None

            # Epoch statistics
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")

                # Early stopping and best-model saving
                if avg_loss < best_loss - 1e-4:
                    best_loss = avg_loss
                    patience_counter = 0
                    self.model.save(str(save_path))
                    print(f"   Saved best model to {save_path}")
                else:
                    patience_counter += 1
                    print(f"   No improvement. Patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print("   Early stopping triggered")
                        break

                # Epoch-end checkpoint (step=0 signals epoch boundary)
                self.save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch + 1,    # next epoch to run
                    step=0,
                    optimizer=optimizer,
                    scaler=scaler,
                    best_loss=best_loss,
                    patience_counter=patience_counter,
                    rng_states={
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    },
                )

        print(f"\nTraining complete! Best loss: {best_loss:.4f}")
        print(f"Model saved to: {save_path}")
        return self.model

    # ------------------------------------------------------------------
    # Persistence helpers (used externally)
    # ------------------------------------------------------------------

    def save_model(self, save_path: Path) -> None:
        """
        Save the trained model.

        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        print(f"Model saved to {save_path}")

    def load_model(self, load_path: Path) -> SentenceTransformer:
        """
        Load a trained model.

        Args:
            load_path: Path to load the model from

        Returns:
            Loaded SentenceTransformer model
        """
        self.model = SentenceTransformer(str(load_path), device=self.device)
        print(f"Model loaded from {load_path}")
        return self.model
