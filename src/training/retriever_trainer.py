"""
Retriever training module for fine-tuning sentence transformers.

This module handles training the retriever model using contrastive learning
on question-passage pairs from HotpotQA.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
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
        
        # Move model to device
        self.model.to(self.device)
        
        # Set max sequence length
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = min(
                self.model.max_seq_length,
                config.retriever_max_seq_length
            )
        
        print(f"Retriever trainer initialized on {self.device}")
    
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
        
        # Convert to InputExample format for sentence-transformers
        input_examples = [
            InputExample(texts=[ex.question, ex.positive_passage])
            for ex in examples
        ]
        
        dataloader = DataLoader(
            input_examples,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            collate_fn=lambda x: x,  # Return list of InputExample
        )
        
        print(f"Created DataLoader: {len(input_examples)} examples, batch_size={batch_size}")
        return dataloader
    
    def train(
        self,
        train_loader: DataLoader,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        patience: Optional[int] = None,
        temperature: Optional[float] = None,
        accumulation_steps: Optional[int] = None,
        use_fp16: Optional[bool] = None,
        save_path: Optional[Path] = None
    ) -> SentenceTransformer:
        """
        Train the retriever model.
        
        Uses contrastive learning with in-batch negatives:
        - For each batch, positive pairs are on the diagonal
        - All other pairs are treated as negatives
        
        Args:
            train_loader: DataLoader with training examples
            epochs: Number of training epochs
            lr: Learning rate
            patience: Early stopping patience
            temperature: Temperature scaling for similarity scores
            accumulation_steps: Gradient accumulation steps
            use_fp16: Whether to use mixed precision training
            save_path: Path to save best model (default: from config)
            
        Returns:
            Trained SentenceTransformer model
        """
        # Use config values if not provided
        epochs = epochs or self.config.retriever_epochs
        lr = lr or self.config.retriever_lr
        patience = patience or self.config.retriever_patience
        temperature = temperature or self.config.retriever_temperature
        accumulation_steps = accumulation_steps or self.config.retriever_accumulation_steps
        use_fp16 = use_fp16 if use_fp16 is not None else self.config.retriever_use_fp16
        save_path = save_path or self.config.get_retriever_model_path()
        
        # Setup training
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=use_fp16)
        
        best_loss = float("inf")
        patience_counter = 0
        
        print(f"\nStarting retriever training...")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {lr}")
        print(f"   Temperature: {temperature}")
        print(f"   Accumulation steps: {accumulation_steps}")
        print(f"   Mixed precision: {use_fp16}")
        
        # Training loop
        for epoch in range(epochs):
            epoch_losses = []
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            optimizer.zero_grad(set_to_none=True)
            
            for step, batch in enumerate(progress):
                # Extract questions and passages
                questions = [ex.texts[0] for ex in batch]
                passages = [ex.texts[1] for ex in batch]
                
                # Tokenize
                q_features = self.model.tokenize(questions)
                p_features = self.model.tokenize(passages)
                
                # Move to device
                q_features = {k: v.to(self.device) for k, v in q_features.items()}
                p_features = {k: v.to(self.device) for k, v in p_features.items()}
                
                # Forward pass with mixed precision
                with autocast(enabled=use_fp16):
                    # Encode questions and passages
                    q_emb = self.model(q_features)["sentence_embedding"]
                    p_emb = self.model(p_features)["sentence_embedding"]
                    
                    # Compute similarity matrix (batch_size x batch_size)
                    # Diagonal elements are positive pairs
                    sim = torch.matmul(q_emb, p_emb.T) * temperature
                    
                    # Labels: diagonal elements (positive pairs)
                    labels = torch.arange(sim.size(0)).to(self.device)
                    
                    # Contrastive loss
                    loss = loss_fn(sim, labels) / accumulation_steps
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                
                # Track loss
                epoch_losses.append(loss.item() * accumulation_steps)
                progress.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})
            
            # Epoch statistics
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")
            
            # Early stopping and model saving
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                
                # Save best model
                self.model.save(str(save_path))
                print(f"Saved best model to {save_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        print(f"\nTraining complete! Best loss: {best_loss:.4f}")
        print(f"Model saved to: {save_path}")
        
        return self.model
    
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
