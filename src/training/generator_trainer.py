"""
Generator training module for fine-tuning seq2seq models.

This module handles training the generator model (e.g., T5, FLAN-T5)
for question answering with retrieved contexts.
"""

import math
from pathlib import Path
from typing import List, Optional, Callable

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

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
        
        # Move model to device
        self.model.to(self.device)
        
        print(f"Generator trainer initialized on {self.device}")
    
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
            collate_fn=lambda x: x,  # Return list of TrainExample
        )
        
        print(f"Created DataLoader: {len(examples)} examples, batch_size={batch_size}")
        return dataloader
    
    def build_prompt(self, question: str, contexts: List[str]) -> str:
        """
        Build input prompt from question and contexts.
        
        Args:
            question: The question
            contexts: List of context passages
            
        Returns:
            Formatted prompt string
        """
        # Format contexts
        context_str = "\n".join(f"- {ctx}" for ctx in contexts)
        
        # Use template from config
        prompt = self.config.qa_prompt_template.format(
            context=context_str,
            question=question
        )
        
        return prompt
    
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
        save_path: Optional[Path] = None
    ) -> AutoModelForSeq2SeqLM:
        """
        Train the generator model.
        
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
            
        Returns:
            Trained model
        """
        # Use config values if not provided
        epochs = epochs or self.config.generator_epochs
        lr = lr or self.config.generator_lr
        warmup_ratio = warmup_ratio or self.config.generator_warmup_ratio
        gradient_accumulation = gradient_accumulation or self.config.generator_gradient_accumulation
        max_input_tokens = max_input_tokens or self.config.generator_max_input_tokens
        max_target_tokens = max_target_tokens or self.config.generator_max_target_tokens
        top_k = top_k or self.config.top_k
        save_path = save_path or self.config.get_generator_model_path()
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=lr)
        
        total_steps = math.ceil(len(train_loader) * epochs / gradient_accumulation)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )
        
        print(f"\nStarting generator training...")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {lr}")
        print(f"   Warmup ratio: {warmup_ratio}")
        print(f"   Gradient accumulation: {gradient_accumulation}")
        print(f"   Max input tokens: {max_input_tokens}")
        print(f"   Max target tokens: {max_target_tokens}")
        
        # Training loop
        self.model.train()
        global_step = 0
        
        for epoch in range(epochs):
            epoch_losses = []
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, batch in enumerate(progress):
                # Prepare batch inputs and targets
                batch_inputs = []
                batch_targets = []
                
                for example in batch:
                    # Get contexts (either pre-computed or retrieve)
                    if example.contexts:
                        contexts = list(example.contexts)
                    elif self.retrieval_fn:
                        contexts = self.retrieval_fn(example.question, top_k)
                    else:
                        raise ValueError(
                            "Either pre-compute contexts or provide retrieval_fn"
                        )
                    
                    # Build prompt
                    prompt = self.build_prompt(example.question, contexts)
                    batch_inputs.append(prompt)
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
                
                # Replace padding token id with -100 (ignored in loss)
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                # Forward pass
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss / gradient_accumulation
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (global_step + 1) % gradient_accumulation == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Track loss
                epoch_losses.append(loss.item() * gradient_accumulation)
                progress.set_postfix({"loss": f"{loss.item() * gradient_accumulation:.4f}"})
                
                global_step += 1
            
            # Epoch statistics
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")
        
        # Save trained model
        self.model.eval()
        self.save_model(save_path)
        
        print(f"\nTraining complete!")
        print(f"Model saved to: {save_path}")
        
        return self.model
    
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
