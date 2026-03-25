"""
LLM-based filtering for RAG quality improvement.

This module implements two-stage filtering:
1. Context Filtering (Pre-Generation): Filter irrelevant retrieved passages
2. Answer Filtering (Post-Generation): Filter low-quality generated answers
"""

import asyncio
import concurrent.futures
import os
import re
import time
from asyncio import Semaphore
from dataclasses import dataclass
from typing import List, Optional, Tuple


def _run_async(coro):
    """
    Run an async coroutine from synchronous code in any context.

    ``asyncio.run()`` raises RuntimeError when called from inside a running
    event loop (e.g. Jupyter notebooks). This helper detects that situation
    and instead submits the coroutine to a fresh event loop running in a
    background thread, which is always safe.
    """
    try:
        asyncio.get_running_loop()
        # We are inside a running loop (Jupyter / IPython).
        # Run the coroutine in a separate thread that owns its own event loop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop — standard asyncio.run() is fine.
        return asyncio.run(coro)

import google.generativeai as genai
from tqdm.auto import tqdm


@dataclass
class ContextFilterResult:
    """Result of context filtering."""
    passage: str
    score: float
    is_relevant: bool
    reasoning: str


@dataclass
class AnswerFilterResult:
    """Result of answer filtering."""
    answer: str
    faithfulness_score: float
    relevance_score: float
    completeness_score: float
    overall_quality: str  # "GOOD" or "BAD"
    reasoning: str


class ContextFilter:
    """
    Filter retrieved contexts before generation using LLM scoring.
    
    Evaluates each retrieved passage for relevance to the question.
    """
    
    PROMPT_TEMPLATE = """You are an expert evaluator for information retrieval systems.

Question: {question}
Context: {passage}

Evaluate if this context is relevant and helpful for answering the question.

Provide your evaluation in this exact format:
Score: [0-10]
Reasoning: [brief explanation in one sentence]

Be strict - only highly relevant contexts should score above 6."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        threshold: float = 6.0,
        max_concurrent: int = 10
    ):
        """
        Initialize context filter.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model_name: Gemini model to use
            threshold: Minimum score for passage to be considered relevant
            max_concurrent: Maximum concurrent API requests
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.threshold = threshold
        self.max_concurrent = max_concurrent
    
    def _parse_response(self, response_text: str) -> Tuple[float, str]:
        """
        Parse LLM response to extract score and reasoning.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Tuple of (score, reasoning)
        """
        score = 0.0
        reasoning = ""
        
        # Extract score
        score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
        
        # Extract reasoning
        reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        return score, reasoning
    
    async def _evaluate_passage_async(
        self,
        question: str,
        passage: str,
        semaphore: Semaphore
    ) -> ContextFilterResult:
        """
        Asynchronously evaluate a single passage.
        
        Args:
            question: The question
            passage: The passage to evaluate
            semaphore: Semaphore for rate limiting
            
        Returns:
            ContextFilterResult
        """
        async with semaphore:
            try:
                prompt = self.PROMPT_TEMPLATE.format(
                    question=question,
                    passage=passage[:1000]  # Limit passage length
                )
                
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate_content(prompt)
                )
                
                score, reasoning = self._parse_response(response.text)
                is_relevant = score >= self.threshold
                
                return ContextFilterResult(
                    passage=passage,
                    score=score,
                    is_relevant=is_relevant,
                    reasoning=reasoning
                )
            
            except Exception as e:
                print(f"Error evaluating passage: {e}")
                # Default to keeping passage on error
                return ContextFilterResult(
                    passage=passage,
                    score=self.threshold,
                    is_relevant=True,
                    reasoning=f"Error: {str(e)}"
                )
    
    async def filter_contexts_async(
        self,
        question: str,
        passages: List[str]
    ) -> List[ContextFilterResult]:
        """
        Filter contexts asynchronously.
        
        Args:
            question: The question
            passages: List of retrieved passages
            
        Returns:
            List of ContextFilterResult objects
        """
        semaphore = Semaphore(self.max_concurrent)
        
        tasks = [
            self._evaluate_passage_async(question, passage, semaphore)
            for passage in passages
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    def filter_contexts(
        self,
        question: str,
        passages: List[str]
    ) -> List[ContextFilterResult]:
        """
        Filter contexts (synchronous wrapper).
        
        Args:
            question: The question
            passages: List of retrieved passages
            
        Returns:
            List of ContextFilterResult objects
        """
        return _run_async(self.filter_contexts_async(question, passages))
    
    def get_filtered_passages(
        self,
        question: str,
        passages: List[str]
    ) -> List[str]:
        """
        Get only the relevant passages after filtering.
        
        Args:
            question: The question
            passages: List of retrieved passages
            
        Returns:
            List of relevant passages
        """
        results = self.filter_contexts(question, passages)
        return [r.passage for r in results if r.is_relevant]


class AnswerFilter:
    """
    Filter generated answers after generation using LLM evaluation.
    
    Evaluates answers on faithfulness, relevance, and completeness.
    """
    
    PROMPT_TEMPLATE = """You are an expert evaluator for question-answering systems.

Question: {question}
Generated Answer: {answer}
Retrieved Context: {context}

Evaluate this answer on three criteria (0-10 scale):

1. Faithfulness: Is the answer grounded in the provided context? No hallucinations?
2. Relevance: Does the answer directly address the question?
3. Completeness: Is the answer thorough and comprehensive for a long-form response?

Provide your evaluation in this exact format:
Faithfulness: [0-10]
Relevance: [0-10]
Completeness: [0-10]
Overall: [GOOD or BAD]
Reasoning: [brief explanation in 1-2 sentences]

Be strict - answers should score high on all three criteria to be marked GOOD."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        threshold: float = 6.0,
        max_concurrent: int = 10
    ):
        """
        Initialize answer filter.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model_name: Gemini model to use
            threshold: Minimum average score for answer to be considered good
            max_concurrent: Maximum concurrent API requests
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.threshold = threshold
        self.max_concurrent = max_concurrent
    
    def _parse_response(self, response_text: str) -> Tuple[float, float, float, str, str]:
        """
        Parse LLM response to extract scores and quality.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Tuple of (faithfulness, relevance, completeness, overall, reasoning)
        """
        faithfulness = 0.0
        relevance = 0.0
        completeness = 0.0
        overall = "BAD"
        reasoning = ""
        
        # Extract scores
        faith_match = re.search(r'Faithfulness:\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
        if faith_match:
            faithfulness = float(faith_match.group(1))
        
        rel_match = re.search(r'Relevance:\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
        if rel_match:
            relevance = float(rel_match.group(1))
        
        comp_match = re.search(r'Completeness:\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
        if comp_match:
            completeness = float(comp_match.group(1))
        
        # Extract overall quality
        overall_match = re.search(r'Overall:\s*(GOOD|BAD)', response_text, re.IGNORECASE)
        if overall_match:
            overall = overall_match.group(1).upper()
        
        # Extract reasoning
        reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        return faithfulness, relevance, completeness, overall, reasoning
    
    async def _evaluate_answer_async(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        semaphore: Semaphore
    ) -> AnswerFilterResult:
        """
        Asynchronously evaluate a single answer.
        
        Args:
            question: The question
            answer: The generated answer
            contexts: Retrieved contexts used for generation
            semaphore: Semaphore for rate limiting
            
        Returns:
            AnswerFilterResult
        """
        async with semaphore:
            try:
                # Combine contexts
                context_str = "\n\n".join([f"[{i+1}] {ctx[:500]}" for i, ctx in enumerate(contexts[:5])])
                
                prompt = self.PROMPT_TEMPLATE.format(
                    question=question,
                    answer=answer[:1000],  # Limit answer length
                    context=context_str
                )
                
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate_content(prompt)
                )
                
                faithfulness, relevance, completeness, overall, reasoning = self._parse_response(response.text)
                
                return AnswerFilterResult(
                    answer=answer,
                    faithfulness_score=faithfulness,
                    relevance_score=relevance,
                    completeness_score=completeness,
                    overall_quality=overall,
                    reasoning=reasoning
                )
            
            except Exception as e:
                print(f"Error evaluating answer: {e}")
                # Default to BAD on error to be conservative
                return AnswerFilterResult(
                    answer=answer,
                    faithfulness_score=0.0,
                    relevance_score=0.0,
                    completeness_score=0.0,
                    overall_quality="BAD",
                    reasoning=f"Error: {str(e)}"
                )
    
    async def filter_answers_async(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]]
    ) -> List[AnswerFilterResult]:
        """
        Filter answers asynchronously.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts_list: List of context lists (one per question)
            
        Returns:
            List of AnswerFilterResult objects
        """
        semaphore = Semaphore(self.max_concurrent)
        
        tasks = [
            self._evaluate_answer_async(q, a, c, semaphore)
            for q, a, c in zip(questions, answers, contexts_list)
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    def filter_answers(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]]
    ) -> List[AnswerFilterResult]:
        """
        Filter answers (synchronous wrapper).
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts_list: List of context lists (one per question)
            
        Returns:
            List of AnswerFilterResult objects
        """
        return _run_async(self.filter_answers_async(questions, answers, contexts_list))
    
    def evaluate_single_answer(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> AnswerFilterResult:
        """
        Evaluate a single answer.
        
        Args:
            question: The question
            answer: The generated answer
            contexts: Retrieved contexts
            
        Returns:
            AnswerFilterResult
        """
        results = self.filter_answers([question], [answer], [contexts])
        return results[0]


class LLMFilterPipeline:
    """
    Complete two-stage filtering pipeline: context + answer filtering.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        context_threshold: float = 6.0,
        answer_threshold: float = 6.0,
        max_concurrent: int = 10
    ):
        """
        Initialize filtering pipeline.
        
        Args:
            api_key: Google API key
            model_name: Gemini model to use
            context_threshold: Threshold for context filtering
            answer_threshold: Threshold for answer filtering
            max_concurrent: Maximum concurrent API requests
        """
        self.context_filter = ContextFilter(
            api_key=api_key,
            model_name=model_name,
            threshold=context_threshold,
            max_concurrent=max_concurrent
        )
        
        self.answer_filter = AnswerFilter(
            api_key=api_key,
            model_name=model_name,
            threshold=answer_threshold,
            max_concurrent=max_concurrent
        )
    
    def filter_contexts(
        self,
        question: str,
        passages: List[str]
    ) -> Tuple[List[str], List[ContextFilterResult]]:
        """
        Filter contexts before generation.
        
        Args:
            question: The question
            passages: Retrieved passages
            
        Returns:
            Tuple of (filtered_passages, filter_results)
        """
        results = self.context_filter.filter_contexts(question, passages)
        filtered_passages = [r.passage for r in results if r.is_relevant]
        return filtered_passages, results
    
    def filter_answer(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> AnswerFilterResult:
        """
        Filter answer after generation.
        
        Args:
            question: The question
            answer: Generated answer
            contexts: Contexts used for generation
            
        Returns:
            AnswerFilterResult
        """
        return self.answer_filter.evaluate_single_answer(question, answer, contexts)
    
    def process_batch(
        self,
        questions: List[str],
        retrieved_passages_list: List[List[str]],
        generated_answers: List[str],
        show_progress: bool = True
    ) -> Tuple[List[AnswerFilterResult], List[List[ContextFilterResult]]]:
        """
        Process a batch of questions with both filtering stages.
        
        Args:
            questions: List of questions
            retrieved_passages_list: List of retrieved passage lists
            generated_answers: List of generated answers
            show_progress: Show progress bar
            
        Returns:
            Tuple of (answer_results, context_results_list)
        """
        # Stage 1: Filter contexts (not used for already-generated answers, but for analysis)
        context_results_list = []
        if show_progress:
            iterator = tqdm(
                zip(questions, retrieved_passages_list),
                total=len(questions),
                desc="Filtering contexts"
            )
        else:
            iterator = zip(questions, retrieved_passages_list)
        
        for question, passages in iterator:
            results = self.context_filter.filter_contexts(question, passages)
            context_results_list.append(results)
        
        # Stage 2: Filter answers
        answer_results = self.answer_filter.filter_answers(
            questions,
            generated_answers,
            retrieved_passages_list
        )
        
        return answer_results, context_results_list
