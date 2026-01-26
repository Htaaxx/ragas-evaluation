"""
Question Answering pipeline combining retrieval and generation.

This module provides the end-to-end RAG pipeline that:
1. Retrieves relevant documents for a question
2. Generates an answer using the retrieved contexts
"""

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..config import RAGConfig
from .indexer import DocumentIndexer


class QAPipeline:
    """
    End-to-end Question Answering pipeline using RAG.
    
    Pipeline:
    1. Encode question with retriever
    2. Search FAISS index for relevant documents
    3. Format prompt with question + contexts
    4. Generate answer with seq2seq model
    """
    
    def __init__(
        self,
        encoder: SentenceTransformer,
        generator: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        indexer: DocumentIndexer,
        config: RAGConfig,
        device: Optional[str] = None
    ):
        """
        Initialize the QA pipeline.
        
        Args:
            encoder: Retriever model
            generator: Generator model
            tokenizer: Tokenizer for generator
            indexer: Document indexer with FAISS index
            config: Configuration object
            device: Device to use
        """
        self.encoder = encoder
        self.generator = generator
        self.tokenizer = tokenizer
        self.indexer = indexer
        self.config = config
        self.device = device or config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move models to device
        self.encoder.to(self.device)
        self.generator.to(self.device)
        self.generator.eval()
        
        print(f"QA Pipeline initialized on {self.device}")
    
    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve relevant documents for a question.
        
        Args:
            question: The question
            top_k: Number of documents to retrieve
            
        Returns:
            Tuple of (documents, scores)
        """
        top_k = top_k or self.config.top_k
        documents, scores, _ = self.indexer.search(question, top_k=top_k)
        return documents, scores
    
    def build_prompt(self, question: str, contexts: List[str]) -> str:
        """
        Build prompt from question and contexts.
        
        Args:
            question: The question
            contexts: List of context passages
            
        Returns:
            Formatted prompt
        """
        context_str = "\n".join(f"- {ctx}" for ctx in contexts)
        prompt = self.config.qa_prompt_template.format(
            context=context_str,
            question=question
        )
        return prompt
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None
    ) -> str:
        """
        Generate answer from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated answer
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.generation_temperature
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.generator_max_input_tokens
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample and temperature is not None,
                temperature=temperature if temperature else 1.0,
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_contexts: bool = False
    ) -> str | Tuple[str, List[str]]:
        """
        Answer a question using the full RAG pipeline.
        
        Args:
            question: The question
            top_k: Number of documents to retrieve
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_contexts: Whether to return retrieved contexts
            
        Returns:
            Generated answer, or (answer, contexts) if return_contexts=True
        """
        # Step 1: Retrieve contexts
        contexts, scores = self.retrieve(question, top_k=top_k)
        
        # Step 2: Build prompt
        prompt = self.build_prompt(question, contexts)
        
        # Step 3: Generate answer
        answer = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        if return_contexts:
            return answer, contexts
        return answer
    
    def batch_answer(
        self,
        questions: List[str],
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        show_progress: bool = True
    ) -> List[str]:
        """
        Answer multiple questions.
        
        Args:
            questions: List of questions
            top_k: Number of documents to retrieve per question
            max_new_tokens: Maximum tokens to generate
            show_progress: Whether to show progress bar
            
        Returns:
            List of generated answers
        """
        from tqdm import tqdm
        
        answers = []
        iterator = questions
        
        if show_progress:
            iterator = tqdm(questions, desc="Answering questions")
        
        for question in iterator:
            answer = self.answer(
                question,
                top_k=top_k,
                max_new_tokens=max_new_tokens
            )
            answers.append(answer)
        
        return answers
    
    @classmethod
    def from_pretrained(
        cls,
        encoder_path: Path,
        generator_path: Path,
        index_path: Path,
        config: RAGConfig,
        device: Optional[str] = None
    ) -> "QAPipeline":
        """
        Load a complete QA pipeline from pretrained models.
        
        Args:
            encoder_path: Path to trained encoder
            generator_path: Path to trained generator
            index_path: Path to FAISS index
            config: Configuration object
            device: Device to use
            
        Returns:
            Loaded QA pipeline
        """
        device = device or config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading QA pipeline from pretrained models...")
        
        # Load encoder
        print(f"   Loading encoder from {encoder_path}...")
        encoder = SentenceTransformer(str(encoder_path), device=device)
        
        # Load generator
        print(f"   Loading generator from {generator_path}...")
        generator = AutoModelForSeq2SeqLM.from_pretrained(generator_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(generator_path)
        
        # Load index
        print(f"   Loading index from {index_path}...")
        indexer = DocumentIndexer(encoder, config)
        indexer.load_index(index_path)
        
        print("QA pipeline loaded successfully!")
        
        return cls(
            encoder=encoder,
            generator=generator,
            tokenizer=tokenizer,
            indexer=indexer,
            config=config,
            device=device
        )
