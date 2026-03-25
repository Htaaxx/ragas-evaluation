"""
Main RAG System class that orchestrates all components.

This module provides a high-level interface for:
- Loading and caching models
- Training retriever and generator
- Building indices
- Question answering
"""

from pathlib import Path
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import RAGConfig
from .data.loader import HotpotQALoader
from .evaluation.retriever_evaluator import RetrieverEvaluator
from .retrieval.indexer import DocumentIndexer
from .retrieval.qa_pipeline import QAPipeline
from .training.generator_trainer import GeneratorTrainer
from .training.retriever_trainer import RetrieverTrainer
from .utils.model_cache import ModelCache, disable_hf_repo_templates


class RAGSystem:
    """
    Complete RAG system for training and inference.
    
    Features:
    - Automatic model downloading and caching
    - Retriever training with contrastive learning
    - Generator training with retrieved contexts
    - FAISS indexing for efficient retrieval
    - End-to-end question answering
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        encoder_model: Optional[str] = None,
        generator_model: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the RAG system.
        
        Args:
            config: Configuration object (creates default if None)
            encoder_model: Override encoder model from config
            generator_model: Override generator model from config
            device: Override device from config
        """
        # Setup configuration
        self.config = config or RAGConfig()
        
        if encoder_model:
            self.config.encoder_model = encoder_model
        if generator_model:
            self.config.generator_model = generator_model
        if device:
            self.config.device = device
        
        # Auto-detect device
        if self.config.device is None:
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\n{'='*60}")
        print(f"Initializing RAG System")
        print(f"{'='*60}")
        print(f"Device: {self.config.device}")
        print(f"Encoder: {self.config.encoder_model}")
        print(f"Generator: {self.config.generator_model}")
        print(f"Models directory: {self.config.models_dir}")
        print(f"Output directory: {self.config.output_dir}")
        
        # Disable HF repo templates check
        disable_hf_repo_templates()
        
        # Initialize model cache
        self.model_cache = ModelCache(
            cache_dir=self.config.models_dir,
            hf_token=self.config.hf_token
        )
        
        # Load models
        self._load_models()
        
        # Initialize components
        self.data_loader = HotpotQALoader()
        self.indexer = DocumentIndexer(self.encoder, self.config)
        
        print(f"{'='*60}\n")
    
    def _load_models(self):
        """Load encoder and generator models with caching."""
        print("\nLoading models...")
        
        # Load encoder
        encoder_path = self.model_cache.load_or_download(self.config.encoder_model)
        self.encoder = SentenceTransformer(str(encoder_path), device=self.config.device)
        
        # Load generator
        generator_path = self.model_cache.load_or_download(self.config.generator_model)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(
            str(generator_path),
            token=self.config.hf_token
        ).to(self.config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(generator_path),
            token=self.config.hf_token
        )
        
        print("Models loaded successfully!")
    
    def load_data(
        self,
        train_path: Optional[str] = None,
        valid_path: Optional[str] = None
    ):
        """
        Load HotpotQA data.
        
        Args:
            train_path: Path to training CSV (default: from config)
            valid_path: Path to validation CSV (default: from config)
        """
        train_path = train_path or self.config.train_data_path
        valid_path = valid_path or self.config.valid_data_path
        
        self.data_loader.load_data(
            train_path=train_path,
            valid_path=valid_path,
            max_train_samples=self.config.max_train_samples,
            max_valid_samples=self.config.max_valid_samples
        )
        
        # Print statistics
        stats = self.data_loader.get_statistics()
        print(f"\nDataset Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    def train_retriever(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        resume_from_checkpoint: bool = True,
    ):
        """
        Train the retriever model.

        Args:
            epochs: Number of epochs (default: from config)
            batch_size: Batch size (default: from config)
            lr: Learning rate (default: from config)
            resume_from_checkpoint: Resume from the last checkpoint if one exists
        """
        print(f"\n{'='*60}")
        print("Training Retriever")
        print(f"{'='*60}")

        examples = self.data_loader.create_retriever_examples()

        # Store trainer on self so callers can free its memory after training
        self.retriever_trainer = RetrieverTrainer(
            model=self.encoder,
            config=self.config,
            device=self.config.device
        )

        train_loader = self.retriever_trainer.create_dataloader(
            examples=examples,
            batch_size=batch_size,
            shuffle=True
        )

        self.encoder = self.retriever_trainer.train(
            train_loader=train_loader,
            epochs=epochs,
            lr=lr,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        print(f"{'='*60}\n")
    
    def evaluate_retriever(self):
        """Evaluate the retriever on validation set."""
        print(f"\n{'='*60}")
        print("Evaluating Retriever")
        print(f"{'='*60}")
        
        evaluator = RetrieverEvaluator(
            encoder=self.encoder,
            config=self.config,
            device=self.config.device
        )
        
        cache_path = self.config.output_dir / f"{self.config.eval_cache_name}.pt"
        
        metrics = evaluator.evaluate(
            df_valid=self.data_loader.df_valid,
            cache_path=cache_path,
            rebuild_cache=self.config.eval_rebuild_cache
        )
        
        print(f"{'='*60}\n")
        return metrics
    
    def build_index(self):
        """Build FAISS index from corpus."""
        print(f"\n{'='*60}")
        print("Building FAISS Index")
        print(f"{'='*60}")
        
        # Build corpus
        corpus_texts, doc_titles = self.data_loader.build_corpus()
        
        # Build index
        self.indexer.build_index(
            documents=corpus_texts,
            titles=doc_titles
        )
        
        # Save index
        self.indexer.save_index()
        
        print(f"{'='*60}\n")
    
    def train_generator(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        max_examples: Optional[int] = None,
        resume_from_checkpoint: bool = True,
    ):
        """
        Train the generator model.

        Args:
            epochs: Number of epochs (default: from config)
            batch_size: Batch size (default: from config)
            lr: Learning rate (default: from config)
            max_examples: Maximum training examples (None = all)
            resume_from_checkpoint: Resume from the last checkpoint if one exists
        """
        print(f"\n{'='*60}")
        print("Training Generator")
        print(f"{'='*60}")

        examples = self.data_loader.create_generator_examples(
            max_examples=max_examples
        )

        def retrieval_fn(question: str, top_k: int) -> List[str]:
            docs, _, _ = self.indexer.search(question, top_k=top_k)
            return docs

        trainer = GeneratorTrainer(
            model=self.generator,
            tokenizer=self.tokenizer,
            config=self.config,
            retrieval_fn=retrieval_fn,
            device=self.config.device
        )

        train_loader = trainer.create_dataloader(
            examples=examples,
            batch_size=batch_size,
            shuffle=True
        )

        self.generator = trainer.train(
            train_loader=train_loader,
            epochs=epochs,
            lr=lr,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        print(f"{'='*60}\n")
    
    def create_qa_pipeline(self) -> QAPipeline:
        """
        Create a QA pipeline for inference.
        
        Returns:
            QAPipeline instance
        """
        return QAPipeline(
            encoder=self.encoder,
            generator=self.generator,
            tokenizer=self.tokenizer,
            indexer=self.indexer,
            config=self.config,
            device=self.config.device
        )
    
    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_contexts: bool = False
    ):
        """
        Answer a question.
        
        Args:
            question: The question
            top_k: Number of documents to retrieve
            return_contexts: Whether to return contexts
            
        Returns:
            Answer or (answer, contexts) if return_contexts=True
        """
        qa_pipeline = self.create_qa_pipeline()
        return qa_pipeline.answer(
            question=question,
            top_k=top_k,
            return_contexts=return_contexts
        )
    
    @classmethod
    def from_pretrained(
        cls,
        encoder_path: Path,
        generator_path: Path,
        index_path: Path,
        config: Optional[RAGConfig] = None
    ) -> "RAGSystem":
        """
        Load a pretrained RAG system.
        
        Args:
            encoder_path: Path to trained encoder
            generator_path: Path to trained generator
            index_path: Path to FAISS index
            config: Configuration (creates default if None)
            
        Returns:
            Loaded RAG system
        """
        config = config or RAGConfig()
        
        print(f"\n{'='*60}")
        print("Loading Pretrained RAG System")
        print(f"{'='*60}")
        
        # Create system (will load base models)
        system = cls(config=config)
        
        # Load trained encoder
        print(f"Loading trained encoder from {encoder_path}...")
        system.encoder = SentenceTransformer(str(encoder_path), device=config.device)
        
        # Load trained generator
        print(f"Loading trained generator from {generator_path}...")
        system.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_path).to(config.device)
        system.tokenizer = AutoTokenizer.from_pretrained(generator_path)
        
        # Load index
        print(f"Loading index from {index_path}...")
        system.indexer = DocumentIndexer(system.encoder, config)
        system.        indexer.load_index(index_path)
        
        print(f"{'='*60}\n")
        print("Pretrained RAG system loaded successfully!")
        
        return system
