"""
LLM-based filtering for RAG quality improvement.

Two-stage filtering:
1. Context Filtering (Pre-Generation): Filter irrelevant retrieved passages
2. Answer Filtering (Post-Generation): Filter low-quality generated answers
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import re
from asyncio import Semaphore
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import google.generativeai as genai
from tqdm.auto import tqdm

from ..configs import load_config

logger = logging.getLogger(__name__)

# Load filtering limits from config; fall back to sensible defaults
try:
    _FILTER_CFG = load_config("filtering")
except FileNotFoundError:
    _FILTER_CFG = {}

_CTX_FILTER_CFG = _FILTER_CFG.get("context_filter", {})
_ANS_FILTER_CFG = _FILTER_CFG.get("answer_filter", {})

MAX_PASSAGE_CHARS: int = _CTX_FILTER_CFG.get("max_passage_chars", 1000)
MAX_CONTEXT_CHUNKS: int = _CTX_FILTER_CFG.get("max_context_chunks", 5)
MAX_ANSWER_CHARS: int = _ANS_FILTER_CFG.get("max_answer_chars", 1000)
MAX_CONTEXT_CHARS: int = _ANS_FILTER_CFG.get("max_context_chars", 500)


def _run_async(coro: Any) -> Any:
    """
    Run an async coroutine from synchronous code in any context.

    ``asyncio.run()`` raises RuntimeError when called from inside a running
    event loop (e.g. Jupyter notebooks). This helper detects that situation
    and submits the coroutine to a fresh loop in a background thread.
    """
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ContextFilter
# ---------------------------------------------------------------------------


class ContextFilter:
    """Filter retrieved contexts before generation using LLM scoring."""

    PROMPT_TEMPLATE = (
        "You are an expert evaluator for information retrieval systems.\n\n"
        "Question: {question}\n"
        "Context: {passage}\n\n"
        "Evaluate if this context is relevant and helpful for answering the question.\n\n"
        "Provide your evaluation in this exact format:\n"
        "Score: [0-10]\n"
        "Reasoning: [brief explanation in one sentence]\n\n"
        "Be strict - only highly relevant contexts should score above 6."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        threshold: float = 6.0,
        max_concurrent: int = 10,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.threshold = threshold
        self.max_concurrent = max_concurrent

    def _parse_response(self, response_text: str) -> Tuple[float, str]:
        score = 0.0
        reasoning = ""

        score_match = re.search(
            r"Score:\s*(\d+(?:\.\d+)?)", response_text, re.IGNORECASE
        )
        if score_match:
            score = float(score_match.group(1))

        reasoning_match = re.search(
            r"Reasoning:\s*(.+?)(?:\n|$)", response_text,
            re.IGNORECASE | re.DOTALL,
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return score, reasoning

    async def _evaluate_passage_async(
        self,
        question: str,
        passage: str,
        semaphore: Semaphore,
    ) -> ContextFilterResult:
        async with semaphore:
            try:
                prompt = self.PROMPT_TEMPLATE.format(
                    question=question,
                    passage=passage[:MAX_PASSAGE_CHARS],
                )

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.model.generate_content(prompt)
                )

                score, reasoning = self._parse_response(response.text)
                is_relevant = score >= self.threshold

                return ContextFilterResult(
                    passage=passage,
                    score=score,
                    is_relevant=is_relevant,
                    reasoning=reasoning,
                )

            except Exception as exc:
                logger.warning("Error evaluating passage: %s", exc)
                return ContextFilterResult(
                    passage=passage,
                    score=self.threshold,
                    is_relevant=True,
                    reasoning=f"Error: {exc}",
                )

    async def filter_contexts_async(
        self,
        question: str,
        passages: List[str],
    ) -> List[ContextFilterResult]:
        semaphore = Semaphore(self.max_concurrent)
        tasks = [
            self._evaluate_passage_async(question, p, semaphore)
            for p in passages
        ]
        return list(await asyncio.gather(*tasks))

    def filter_contexts(
        self,
        question: str,
        passages: List[str],
    ) -> List[ContextFilterResult]:
        """Filter contexts (synchronous wrapper)."""
        return _run_async(self.filter_contexts_async(question, passages))

    def get_filtered_passages(
        self,
        question: str,
        passages: List[str],
    ) -> List[str]:
        """Return only the relevant passages after filtering."""
        results = self.filter_contexts(question, passages)
        return [r.passage for r in results if r.is_relevant]


# ---------------------------------------------------------------------------
# AnswerFilter
# ---------------------------------------------------------------------------


class AnswerFilter:
    """Filter generated answers after generation using LLM evaluation."""

    PROMPT_TEMPLATE = (
        "You are an expert evaluator for question-answering systems.\n\n"
        "Question: {question}\n"
        "Generated Answer: {answer}\n"
        "Retrieved Context: {context}\n\n"
        "Evaluate this answer on three criteria (0-10 scale):\n\n"
        "1. Faithfulness: Is the answer grounded in the provided context?\n"
        "2. Relevance: Does the answer directly address the question?\n"
        "3. Completeness: Is the answer thorough for a long-form response?\n\n"
        "Provide your evaluation in this exact format:\n"
        "Faithfulness: [0-10]\n"
        "Relevance: [0-10]\n"
        "Completeness: [0-10]\n"
        "Overall: [GOOD or BAD]\n"
        "Reasoning: [brief explanation in 1-2 sentences]\n\n"
        "Be strict - answers should score high on all three to be GOOD."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        threshold: float = 6.0,
        max_concurrent: int = 10,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.threshold = threshold
        self.max_concurrent = max_concurrent

    def _parse_response(
        self, response_text: str
    ) -> Tuple[float, float, float, str, str]:
        faithfulness = 0.0
        relevance = 0.0
        completeness = 0.0
        overall = "BAD"
        reasoning = ""

        for pattern, setter in [
            (r"Faithfulness:\s*(\d+(?:\.\d+)?)", "faithfulness"),
            (r"Relevance:\s*(\d+(?:\.\d+)?)", "relevance"),
            (r"Completeness:\s*(\d+(?:\.\d+)?)", "completeness"),
        ]:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                locals()[setter]  # noqa — just for readability
                if setter == "faithfulness":
                    faithfulness = float(match.group(1))
                elif setter == "relevance":
                    relevance = float(match.group(1))
                elif setter == "completeness":
                    completeness = float(match.group(1))

        overall_match = re.search(
            r"Overall:\s*(GOOD|BAD)", response_text, re.IGNORECASE
        )
        if overall_match:
            overall = overall_match.group(1).upper()

        reasoning_match = re.search(
            r"Reasoning:\s*(.+?)(?:\n|$)", response_text,
            re.IGNORECASE | re.DOTALL,
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return faithfulness, relevance, completeness, overall, reasoning

    async def _evaluate_answer_async(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        semaphore: Semaphore,
    ) -> AnswerFilterResult:
        async with semaphore:
            try:
                context_str = "\n\n".join(
                    f"[{i + 1}] {ctx[:MAX_CONTEXT_CHARS]}"
                    for i, ctx in enumerate(contexts[:MAX_CONTEXT_CHUNKS])
                )

                prompt = self.PROMPT_TEMPLATE.format(
                    question=question,
                    answer=answer[:MAX_ANSWER_CHARS],
                    context=context_str,
                )

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.model.generate_content(prompt)
                )

                (faithfulness, relevance, completeness,
                 overall, reasoning) = self._parse_response(response.text)

                return AnswerFilterResult(
                    answer=answer,
                    faithfulness_score=faithfulness,
                    relevance_score=relevance,
                    completeness_score=completeness,
                    overall_quality=overall,
                    reasoning=reasoning,
                )

            except Exception as exc:
                logger.warning("Error evaluating answer: %s", exc)
                return AnswerFilterResult(
                    answer=answer,
                    faithfulness_score=0.0,
                    relevance_score=0.0,
                    completeness_score=0.0,
                    overall_quality="BAD",
                    reasoning=f"Error: {exc}",
                )

    async def filter_answers_async(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
    ) -> List[AnswerFilterResult]:
        semaphore = Semaphore(self.max_concurrent)
        tasks = [
            self._evaluate_answer_async(q, a, c, semaphore)
            for q, a, c in zip(questions, answers, contexts_list)
        ]
        return list(await asyncio.gather(*tasks))

    def filter_answers(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
    ) -> List[AnswerFilterResult]:
        """Filter answers (synchronous wrapper)."""
        return _run_async(
            self.filter_answers_async(questions, answers, contexts_list)
        )

    def evaluate_single_answer(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> AnswerFilterResult:
        """Evaluate a single answer."""
        results = self.filter_answers([question], [answer], [contexts])
        return results[0]


# ---------------------------------------------------------------------------
# LLMFilterPipeline
# ---------------------------------------------------------------------------


class LLMFilterPipeline:
    """Complete two-stage filtering pipeline: context + answer filtering."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        context_threshold: float = 6.0,
        answer_threshold: float = 6.0,
        max_concurrent: int = 10,
    ) -> None:
        self.context_filter = ContextFilter(
            api_key=api_key,
            model_name=model_name,
            threshold=context_threshold,
            max_concurrent=max_concurrent,
        )
        self.answer_filter = AnswerFilter(
            api_key=api_key,
            model_name=model_name,
            threshold=answer_threshold,
            max_concurrent=max_concurrent,
        )

    def filter_contexts(
        self,
        question: str,
        passages: List[str],
    ) -> Tuple[List[str], List[ContextFilterResult]]:
        results = self.context_filter.filter_contexts(question, passages)
        filtered_passages = [r.passage for r in results if r.is_relevant]
        return filtered_passages, results

    def filter_answer(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> AnswerFilterResult:
        return self.answer_filter.evaluate_single_answer(question, answer, contexts)

    def process_batch(
        self,
        questions: List[str],
        retrieved_passages_list: List[List[str]],
        generated_answers: List[str],
        show_progress: bool = True,
    ) -> Tuple[List[AnswerFilterResult], List[List[ContextFilterResult]]]:
        """Process a batch with both filtering stages."""
        context_results_list: List[List[ContextFilterResult]] = []
        iterator: Any
        if show_progress:
            iterator = tqdm(
                zip(questions, retrieved_passages_list),
                total=len(questions),
                desc="Filtering contexts",
            )
        else:
            iterator = zip(questions, retrieved_passages_list)

        for question, passages in iterator:
            results = self.context_filter.filter_contexts(question, passages)
            context_results_list.append(results)

        answer_results = self.answer_filter.filter_answers(
            questions, generated_answers, retrieved_passages_list,
        )

        return answer_results, context_results_list
