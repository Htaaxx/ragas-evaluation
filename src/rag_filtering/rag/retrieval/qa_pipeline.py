"""
Question Answering pipeline combining retrieval and generation.

Provides the end-to-end RAG pipeline:
1. Retrieve relevant documents for a question
2. Generate an answer using the retrieved contexts
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from rag_filtering.rag.config import RAGConfig
from rag_filtering.rag.retrieval.indexer import DocumentIndexer

if TYPE_CHECKING:
    from ..filtering.data_models import FilterDecision
    from ..filtering.learned_filter import AnswerQualityClassifier

logger = logging.getLogger(__name__)


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
        device: Optional[str] = None,
    ) -> None:
        self.encoder = encoder
        self.generator = generator
        self.tokenizer = tokenizer
        self.indexer = indexer
        self.config = config
        self.device = (
            device or config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.encoder.to(self.device)
        self.generator.to(self.device)
        self.generator.eval()

        logger.info("QA Pipeline initialized on %s", self.device)

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> Tuple[List[str], List[float]]:
        """Retrieve relevant documents for a question."""
        top_k = top_k or self.config.top_k
        documents, scores, _ = self.indexer.search(question, top_k=top_k)
        return documents, scores

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        """Build prompt from question and contexts.

        The joined context is token-capped so the instruction + question
        (placed first by the template) always fit within the generator's
        input window. Without this, long retrieved passages push the
        question past ``generator_max_input_tokens``; the model then sees
        only context and no question, and tends to echo the passage
        verbatim instead of answering.
        """
        context_str = "\n".join(f"- {ctx}" for ctx in contexts)

        # Reserve room for the instruction, question and scaffolding tokens.
        reserve = 80
        q_ids = self.tokenizer(question, add_special_tokens=False)["input_ids"]
        budget = self.config.generator_max_input_tokens - reserve - len(q_ids)
        if budget > 0:
            ctx_ids = self.tokenizer(context_str, add_special_tokens=False)["input_ids"]
            if len(ctx_ids) > budget:
                context_str = self.tokenizer.decode(
                    ctx_ids[:budget], skip_special_tokens=True
                )

        return self.config.qa_prompt_template.format(
            context=context_str, question=question
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> str:
        """Generate answer from prompt."""
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = (
            temperature if temperature is not None
            else self.config.generation_temperature
        )
        do_sample = (
            do_sample if do_sample is not None
            else self.config.do_sample
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.generator_max_input_tokens,
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": 4,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
        if do_sample and temperature:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.generator.generate(**inputs, **gen_kwargs)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_contexts: bool = False,
    ) -> Union[str, Tuple[str, List[str]]]:
        """Answer a question using the full RAG pipeline."""
        contexts, _ = self.retrieve(question, top_k=top_k)
        prompt = self.build_prompt(question, contexts)
        gen_answer = self.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature
        )

        if return_contexts:
            return gen_answer, contexts
        return gen_answer

    def filtered_answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_gate: Optional[AnswerQualityClassifier] = None,
    ) -> Tuple[str, List[str], Optional[FilterDecision]]:
        """Answer a question, then optionally run the quality filter.

        Returns (answer, contexts, decision).  ``decision`` is ``None``
        when no ``filter_gate`` is provided.
        """
        answer, contexts = self.answer(question, top_k=top_k, return_contexts=True)
        if filter_gate is not None:
            decision = filter_gate.predict(question, answer)
            return answer, contexts, decision
        return answer, contexts, None

    def batch_answer(
        self,
        questions: List[str],
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[str]:
        """Answer multiple questions."""
        from tqdm import tqdm

        answers: List[str] = []
        iterator = tqdm(questions, desc="Answering") if show_progress else questions

        for question in iterator:
            gen_answer = self.answer(
                question, top_k=top_k, max_new_tokens=max_new_tokens
            )
            answers.append(gen_answer)

        return answers

    @classmethod
    def from_pretrained(
        cls,
        encoder_path: Path,
        generator_path: Path,
        index_path: Path,
        config: RAGConfig,
        device: Optional[str] = None,
    ) -> QAPipeline:
        """Load a complete QA pipeline from pretrained models."""
        device = (
            device or config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("Loading QA pipeline from pretrained models …")

        encoder = SentenceTransformer(str(encoder_path), device=device)
        generator = AutoModelForSeq2SeqLM.from_pretrained(
            generator_path
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(generator_path)

        indexer = DocumentIndexer(encoder, config)
        indexer.load_index(index_path)

        logger.info("QA pipeline loaded successfully!")

        return cls(
            encoder=encoder,
            generator=generator,
            tokenizer=tokenizer,
            indexer=indexer,
            config=config,
            device=device,
        )
