"""
Configuration management for RAG training system.

This module contains all configuration parameters for the RAG training pipeline,
including model settings, training hyperparameters, and paths.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RAGConfig:
    """Configuration for RAG training system."""
    
    # ==================== Model Configuration ====================
    # Retriever (Encoder) model
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_embedding_dim: int = 384  # Dimension for MiniLM-L6-v2
    
    # Generator model
    generator_model: str = "google/flan-t5-base"
    
    # Device configuration
    device: Optional[str] = None  # None = auto-detect (cuda if available)
    
    # ==================== Directory Configuration ====================
    # Base directories
    project_root: Path = field(default_factory=lambda: Path.cwd())
    models_dir: Path = field(default_factory=lambda: Path("../models"))
    output_dir: Path = field(default_factory=lambda: Path("./rag_output"))
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    
    # Index directory
    index_dir: Optional[Path] = None  # Will be set to output_dir/index if None
    
    # ==================== Data Configuration ====================
    # HotpotQA dataset paths
    train_data_path: str = "data/hotpot_qa/train.csv"
    valid_data_path: str = "data/hotpot_qa/valid.csv"
    
    # Data processing
    max_train_samples: Optional[int] = None  # None = use all
    max_valid_samples: Optional[int] = None  # None = use all
    
    # ==================== Retriever Training Configuration ====================
    # Training hyperparameters
    retriever_epochs: int = 5
    retriever_batch_size: int = 16
    retriever_lr: float = 2e-5
    retriever_patience: int = 3  # Early stopping patience
    retriever_temperature: float = 20.0  # Temperature scaling for contrastive loss
    retriever_accumulation_steps: int = 2  # Gradient accumulation
    retriever_use_fp16: bool = True  # Mixed precision training
    retriever_max_seq_length: int = 128  # Max sequence length for encoder
    
    # Model saving
    retriever_save_name: str = "retriever_trained"
    
    # ==================== Generator Training Configuration ====================
    # Training hyperparameters
    generator_epochs: int = 3
    generator_batch_size: int = 4
    generator_lr: float = 5e-5
    generator_warmup_ratio: float = 0.1
    generator_gradient_accumulation: int = 1
    generator_max_input_tokens: int = 512
    generator_max_target_tokens: int = 128
    
    # Model saving
    generator_save_name: str = "generator_trained"
    
    # ==================== Indexing Configuration ====================
    # FAISS index settings
    index_batch_size: int = 64  # Batch size for encoding documents
    normalize_embeddings: bool = True  # L2 normalization
    
    # ==================== Retrieval Configuration ====================
    # Retrieval settings
    top_k: int = 5  # Number of documents to retrieve
    
    # ==================== Generation Configuration ====================
    # Generation settings
    max_new_tokens: int = 64
    generation_temperature: Optional[float] = None  # None = greedy decoding
    do_sample: bool = False  # Sampling vs greedy
    
    # ==================== Evaluation Configuration ====================
    # Retriever evaluation
    eval_top_k_values: list = field(default_factory=lambda: [1, 3, 5])
    eval_batch_size: int = 64
    eval_cache_name: str = "corpus_embeds"
    eval_rebuild_cache: bool = False
    
    # ==================== API Configuration ====================
    # HuggingFace API
    hf_token: Optional[str] = field(
        default_factory=lambda: os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    
    # ==================== Logging Configuration ====================
    # Logging settings
    log_level: str = "INFO"
    log_interval: int = 10  # Log every N steps
    save_interval: int = 1000  # Save checkpoint every N steps
    
    # ==================== Prompt Templates ====================
    # Prompt template for QA
    qa_prompt_template: str = (
        "Use the context to answer.\n"
        "Context:\n{context}\n"
        "Question: {question}\n"
        "Answer:"
    )
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Convert string paths to Path objects
        self.models_dir = Path(self.models_dir)
        self.output_dir = Path(self.output_dir)
        self.data_dir = Path(self.data_dir)
        
        # Set index_dir if not provided
        if self.index_dir is None:
            self.index_dir = self.output_dir / "index"
        else:
            self.index_dir = Path(self.index_dir)
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
    
    def get_retriever_model_path(self) -> Path:
        """Get path to trained retriever model."""
        return self.models_dir / self.retriever_save_name
    
    def get_generator_model_path(self) -> Path:
        """Get path to trained generator model."""
        return self.models_dir / self.generator_save_name
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "encoder_model": self.encoder_model,
            "generator_model": self.generator_model,
            "retriever_epochs": self.retriever_epochs,
            "retriever_batch_size": self.retriever_batch_size,
            "retriever_lr": self.retriever_lr,
            "generator_epochs": self.generator_epochs,
            "generator_batch_size": self.generator_batch_size,
            "generator_lr": self.generator_lr,
            "top_k": self.top_k,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "RAGConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


# Default configuration instance
default_config = RAGConfig()
