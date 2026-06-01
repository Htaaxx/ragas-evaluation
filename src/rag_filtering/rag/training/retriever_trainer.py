"""
Retriever training module for fine-tuning sentence transformers.

Handles training the retriever model using contrastive learning on
question-passage pairs. Supports crash-safe sub-epoch checkpointing
with auto-resume.
"""

from __future__ import annotations

import gc
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer
from torch import nn
from torch.optim import AdamW

try:
    from torch.amp import GradScaler, autocast

    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore[assignment]

    _AMP_DEVICE = "cuda"
from torch.utils.data import DataLoader
from tqdm import tqdm

from rag_filtering.rag.config import RAGConfig
from rag_filtering.data.base_loader import RetrieverExample

logger = logging.getLogger(__name__)


class RetrieverTrainer:
    """
    Trainer for fine-tuning retriever models using contrastive learning.

    Supports crash-safe checkpointing: every ``checkpoint_steps`` steps
    the full training state is written to disk.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        config: RAGConfig,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = (
            device or config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = min(
                self.model.max_seq_length,
                config.retriever_max_seq_length,
            )
        logger.info("Retriever trainer initialized on %s", self.device)

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    def create_dataloader(
        self,
        examples: List[RetrieverExample],
        batch_size: Optional[int] = None,
        shuffle: bool = True,
    ) -> DataLoader:
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
            num_workers=0,
            collate_fn=lambda x: x,
        )
        logger.info(
            "Created DataLoader: %d examples, batch_size=%d",
            len(input_examples), batch_size,
        )
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
        """Persist full training state for crash-safe resume."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_ckpt_path = checkpoint_dir / "retriever_ckpt_model"
        self.model.save(str(model_ckpt_path))

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
        logger.info(
            "[ckpt] Saved retriever checkpoint -> epoch=%d, step=%d",
            epoch + 1, step,
        )

    def load_checkpoint(
        self,
        checkpoint_dir: Path,
        optimizer: AdamW,
        scaler: GradScaler,
    ) -> Optional[tuple]:
        """Load training state from a previously saved checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        state_path = checkpoint_dir / "retriever_checkpoint.pt"
        model_ckpt_path = checkpoint_dir / "retriever_ckpt_model"

        if not state_path.exists() or not model_ckpt_path.exists():
            return None

        logger.info("Resuming retriever from checkpoint: %s", state_path)
        state = torch.load(
            state_path, map_location=self.device, weights_only=False
        )

        self.model = SentenceTransformer(
            str(model_ckpt_path), device=self.device
        )
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = min(
                self.model.max_seq_length,
                self.config.retriever_max_seq_length,
            )

        self._detach_mmap_handles()

        optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])

        logger.info(
            "Resumed at epoch=%d, step=%d, best_loss=%.4f",
            state["epoch"] + 1, state["step"], state["best_loss"],
        )
        return (
            state["epoch"],
            state["step"],
            state["best_loss"],
            state["patience_counter"],
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
        patience: Optional[int] = None,
        temperature: Optional[float] = None,
        accumulation_steps: Optional[int] = None,
        use_fp16: Optional[bool] = None,
        save_path: Optional[Path] = None,
        resume_from_checkpoint: bool = True,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_steps: Optional[int] = None,
    ) -> SentenceTransformer:
        """Train the retriever with crash-safe sub-epoch checkpointing."""
        hp = self._resolve_hyperparams(
            epochs, lr, patience, temperature,
            accumulation_steps, use_fp16, save_path,
            checkpoint_dir, checkpoint_steps,
        )

        optimizer = AdamW(self.model.parameters(), lr=hp["lr"])
        loss_fn = nn.CrossEntropyLoss()
        scaler = GradScaler(_AMP_DEVICE, enabled=hp["use_fp16"])

        best_loss = float("inf")
        patience_counter = 0
        start_epoch, start_step = 0, 0
        saved_rng_states: Optional[dict] = None

        if resume_from_checkpoint:
            result = self.load_checkpoint(
                hp["checkpoint_dir"], optimizer, scaler
            )
            if result is not None:
                (start_epoch, start_step, best_loss,
                 patience_counter, saved_rng_states) = result

        self._log_training_config(hp)

        for epoch in range(start_epoch, hp["epochs"]):
            epoch_rng = self._manage_rng(epoch, start_epoch, saved_rng_states)
            if epoch != start_epoch:
                start_step = 0

            epoch_losses = self._run_epoch(
                train_loader, optimizer, loss_fn, scaler,
                epoch, hp, start_step, epoch_rng,
            )

            start_step = 0
            saved_rng_states = None

            if not epoch_losses:
                continue

            avg_loss = float(np.mean(epoch_losses))
            logger.info(
                "Epoch %d/%d: avg_loss=%.4f",
                epoch + 1, hp["epochs"], avg_loss,
            )

            best_loss, patience_counter, should_stop = self._check_early_stop(
                avg_loss, best_loss, patience_counter,
                hp["patience"], hp["save_path"],
            )

            self.save_checkpoint(
                checkpoint_dir=hp["checkpoint_dir"],
                epoch=epoch + 1,
                step=0,
                optimizer=optimizer,
                scaler=scaler,
                best_loss=best_loss,
                patience_counter=patience_counter,
                rng_states=self._capture_rng(),
            )

            if should_stop:
                break

        logger.info("Training complete! Best loss: %.4f", best_loss)
        logger.info("Model saved to: %s", hp["save_path"])
        return self.model

    # ------------------------------------------------------------------
    # Training internals
    # ------------------------------------------------------------------

    def _resolve_hyperparams(self, *args) -> Dict:
        (epochs, lr, patience, temperature,
         accumulation_steps, use_fp16, save_path,
         checkpoint_dir, checkpoint_steps) = args
        return {
            "epochs": epochs or self.config.retriever_epochs,
            "lr": lr or self.config.retriever_lr,
            "patience": patience or self.config.retriever_patience,
            "temperature": temperature or self.config.retriever_temperature,
            "accumulation_steps": (
                accumulation_steps or self.config.retriever_accumulation_steps
            ),
            "use_fp16": (
                use_fp16 if use_fp16 is not None
                else self.config.retriever_use_fp16
            ),
            "save_path": save_path or self.config.get_retriever_model_path(),
            "checkpoint_dir": (
                Path(checkpoint_dir) if checkpoint_dir
                else self.config.get_checkpoint_dir()
            ),
            "checkpoint_steps": (
                checkpoint_steps or self.config.retriever_checkpoint_steps
            ),
        }

    def _log_training_config(self, hp: Dict) -> None:
        logger.info("Starting retriever training …")
        for k, v in hp.items():
            logger.info("   %s: %s", k, v)

    def _run_epoch(
        self,
        train_loader: DataLoader,
        optimizer: AdamW,
        loss_fn: nn.CrossEntropyLoss,
        scaler: GradScaler,
        epoch: int,
        hp: Dict,
        start_step: int,
        epoch_rng: dict,
    ) -> List[float]:
        """Run one training epoch, returns list of per-step losses."""
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: List[float] = []

        progress = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{hp['epochs']}"
        )

        for step, batch in enumerate(progress):
            if step < start_step:
                progress.set_postfix(
                    {"status": f"skipping to step {start_step}"}
                )
                continue

            questions = [ex.texts[0] for ex in batch]
            passages = [ex.texts[1] for ex in batch]

            q_features = self.model.tokenize(questions)
            p_features = self.model.tokenize(passages)
            # Newer sentence-transformers return non-tensor entries (e.g. raw
            # text) in the feature dict; only move actual tensors to device.
            q_features = {
                k: (v.to(self.device) if hasattr(v, "to") else v)
                for k, v in q_features.items()
            }
            p_features = {
                k: (v.to(self.device) if hasattr(v, "to") else v)
                for k, v in p_features.items()
            }

            with autocast(_AMP_DEVICE, enabled=hp["use_fp16"]):
                q_emb = self.model(q_features)["sentence_embedding"]
                p_emb = self.model(p_features)["sentence_embedding"]
                sim = torch.matmul(q_emb, p_emb.T) * hp["temperature"]
                labels = torch.arange(sim.size(0)).to(self.device)
                loss = loss_fn(sim, labels) / hp["accumulation_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % hp["accumulation_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            raw_loss = loss.item() * hp["accumulation_steps"]
            epoch_losses.append(raw_loss)
            progress.set_postfix({"loss": f"{raw_loss:.4f}"})

            if (step + 1) % hp["checkpoint_steps"] == 0:
                self.save_checkpoint(
                    checkpoint_dir=hp["checkpoint_dir"],
                    epoch=epoch,
                    step=step + 1,
                    optimizer=optimizer,
                    scaler=scaler,
                    best_loss=float("inf"),
                    patience_counter=0,
                    rng_states=epoch_rng,
                )

        return epoch_losses

    def _check_early_stop(
        self,
        avg_loss: float,
        best_loss: float,
        patience_counter: int,
        patience: int,
        save_path: Path,
    ) -> tuple[float, int, bool]:
        """Return (best_loss, patience_counter, should_stop)."""
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
            self.model.save(str(save_path))
            logger.info("Saved best model to %s", save_path)
        else:
            patience_counter += 1
            logger.info(
                "No improvement. Patience: %d/%d",
                patience_counter, patience,
            )
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                return best_loss, patience_counter, True

        return best_loss, patience_counter, False

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
        return RetrieverTrainer._capture_rng()

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
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        logger.info("Model saved to %s", save_path)

    def load_model(self, load_path: Path) -> SentenceTransformer:
        self.model = SentenceTransformer(
            str(load_path), device=self.device
        )
        logger.info("Model loaded from %s", load_path)
        return self.model
