"""
Main RAG System class that orchestrates all components.

Provides a high-level interface for:
- Loading and caching models
- Training retriever and generator
- Building indices
- Question answering
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from rag_filtering.rag.config import RAGConfig
from rag_filtering.data.asqa_loader import ASQALoader
from rag_filtering.evaluation.retriever_evaluator import RetrieverEvaluator
from rag_filtering.rag.retrieval.indexer import DocumentIndexer
from rag_filtering.rag.retrieval.qa_pipeline import QAPipeline
from rag_filtering.rag.training.generator_trainer import GeneratorTrainer
from rag_filtering.rag.training.retriever_trainer import RetrieverTrainer
from rag_filtering.utils.model_cache import ModelCache, disable_hf_repo_templates

logger = logging.getLogger(__name__)


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
        device: Optional[str] = None,
    ) -> None:
        self.config = config or RAGConfig()

        if encoder_model:
            self.config.encoder_model = encoder_model
        if generator_model:
            self.config.generator_model = generator_model
        if device:
            self.config.device = device

        if self.config.device is None:
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("=" * 60)
        logger.info("Initializing RAG System")
        logger.info("=" * 60)
        logger.info("Device: %s", self.config.device)
        logger.info("Encoder: %s", self.config.encoder_model)
        logger.info("Generator: %s", self.config.generator_model)
        logger.info("Models directory: %s", self.config.models_dir)
        logger.info("Output directory: %s", self.config.output_dir)

        disable_hf_repo_templates()

        self.model_cache = ModelCache(
            cache_dir=self.config.models_dir,
            hf_token=self.config.hf_token,
        )

        self._load_models()

        self.data_loader = ASQALoader()
        self.indexer = DocumentIndexer(self.encoder, self.config)

        logger.info("=" * 60)

    def _load_models(self) -> None:
        """Load encoder and generator models with caching."""
        logger.info("Loading models …")

        encoder_path = self.model_cache.load_or_download(self.config.encoder_model)
        self.encoder = SentenceTransformer(
            str(encoder_path), device=self.config.device
        )

        generator_path = self.model_cache.load_or_download(self.config.generator_model)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(
            str(generator_path), token=self.config.hf_token
        ).to(self.config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(generator_path), token=self.config.hf_token
        )

        logger.info("Models loaded successfully!")

    def load_data(
        self,
        train_path: Optional[str] = None,
        valid_path: Optional[str] = None,
    ) -> None:
        """Load ASQA data."""
        train_path = train_path or self.config.train_data_path
        valid_path = valid_path or self.config.valid_data_path

        self.data_loader.load_data(
            train_path=train_path,
            valid_path=valid_path,
            max_train_samples=self.config.max_train_samples,
            max_valid_samples=self.config.max_valid_samples,
        )

        stats = self.data_loader.get_statistics()
        logger.info("Dataset Statistics:")
        for key, value in stats.items():
            logger.info("   %s: %s", key, value)

    def train_retriever(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        resume_from_checkpoint: bool = True,
    ) -> None:
        """Train the retriever model."""
        logger.info("=" * 60)
        logger.info("Training Retriever")
        logger.info("=" * 60)

        examples = self.data_loader.create_retriever_examples()

        self.retriever_trainer = RetrieverTrainer(
            model=self.encoder,
            config=self.config,
            device=self.config.device,
        )

        train_loader = self.retriever_trainer.create_dataloader(
            examples=examples, batch_size=batch_size, shuffle=True
        )

        self.encoder = self.retriever_trainer.train(
            train_loader=train_loader,
            epochs=epochs,
            lr=lr,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        logger.info("=" * 60)

    def evaluate_retriever(self) -> dict:
        """Evaluate the retriever on validation set."""
        logger.info("=" * 60)
        logger.info("Evaluating Retriever")
        logger.info("=" * 60)

        evaluator = RetrieverEvaluator(
            encoder=self.encoder,
            config=self.config,
            device=self.config.device,
        )

        cache_path = self.config.output_dir / f"{self.config.eval_cache_name}.pt"

        metrics = evaluator.evaluate(
            df_valid=self.data_loader.df_valid,
            cache_path=cache_path,
            rebuild_cache=self.config.eval_rebuild_cache,
        )

        logger.info("=" * 60)
        return metrics

    def build_index(self) -> None:
        """Build FAISS index from corpus."""
        logger.info("=" * 60)
        logger.info("Building FAISS Index")
        logger.info("=" * 60)

        corpus_texts, doc_titles = self.data_loader.build_corpus()
        self.indexer.build_index(documents=corpus_texts, titles=doc_titles)
        self.indexer.save_index()

        logger.info("=" * 60)

    def train_generator(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        max_examples: Optional[int] = None,
        resume_from_checkpoint: bool = True,
    ) -> None:
        """Train the generator model."""
        logger.info("=" * 60)
        logger.info("Training Generator")
        logger.info("=" * 60)

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
            device=self.config.device,
        )

        train_loader = trainer.create_dataloader(
            examples=examples, batch_size=batch_size, shuffle=True
        )

        self.generator = trainer.train(
            train_loader=train_loader,
            epochs=epochs,
            lr=lr,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        logger.info("=" * 60)

    def create_qa_pipeline(self) -> QAPipeline:
        """Create a QA pipeline for inference."""
        return QAPipeline(
            encoder=self.encoder,
            generator=self.generator,
            tokenizer=self.tokenizer,
            indexer=self.indexer,
            config=self.config,
            device=self.config.device,
        )

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_contexts: bool = False,
    ) -> Union[str, Tuple[str, List[str]]]:
        """Answer a question using the full RAG pipeline."""
        qa_pipeline = self.create_qa_pipeline()
        return qa_pipeline.answer(
            question=question,
            top_k=top_k,
            return_contexts=return_contexts,
        )

    @classmethod
    def from_pretrained(
        cls,
        encoder_path: Path,
        generator_path: Path,
        index_path: Path,
        config: Optional[RAGConfig] = None,
    ) -> RAGSystem:
        """Load a pretrained RAG system."""
        config = config or RAGConfig()

        logger.info("=" * 60)
        logger.info("Loading Pretrained RAG System")
        logger.info("=" * 60)

        system = cls(config=config)

        logger.info("Loading trained encoder from %s …", encoder_path)
        system.encoder = SentenceTransformer(
            str(encoder_path), device=config.device
        )

        logger.info("Loading trained generator from %s …", generator_path)
        system.generator = AutoModelForSeq2SeqLM.from_pretrained(
            generator_path
        ).to(config.device)
        system.tokenizer = AutoTokenizer.from_pretrained(generator_path)

        logger.info("Loading index from %s …", index_path)
        system.indexer = DocumentIndexer(system.encoder, config)
        system.indexer.load_index(index_path)

        logger.info("=" * 60)
        logger.info("Pretrained RAG system loaded successfully!")

        return system
