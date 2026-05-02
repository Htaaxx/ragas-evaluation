"""
LLM-based answer scoring for the black-box RAG pipeline.

Scores generated answers against ground truth using an LLM judge.
No context filtering — all retrieved passages are passed through to
the generator; only the final answer is evaluated.
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

from ..configs import load_config

logger = logging.getLogger(__name__)

try:
    _FILTER_CFG = load_config("filtering")
except FileNotFoundError:
    _FILTER_CFG = {}

_ANS_FILTER_CFG = _FILTER_CFG.get("answer_filter", {})

MAX_ANSWER_CHARS: int = _ANS_FILTER_CFG.get("max_answer_chars", 1000)
MAX_GROUND_TRUTH_CHARS: int = _ANS_FILTER_CFG.get("max_ground_truth_chars", 1000)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from synchronous code in any context."""
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class AnswerScoreResult:
    """Result of scoring a generated answer against ground truth."""

    answer: str
    correctness_score: float
    similarity_score: float
    completeness_score: float
    overall_quality: str  # "GOOD" or "BAD"
    reasoning: str


# ---------------------------------------------------------------------------
# AnswerFilter — scores answers against ground truth
# ---------------------------------------------------------------------------


class AnswerFilter:
    """Score generated answers by comparing them to the ground-truth answer."""

    PROMPT_TEMPLATE = (
        "You are an expert evaluator for question-answering systems.\n\n"
        "Question: {question}\n"
        "Generated Answer: {answer}\n"
        "Correct Answer: {ground_truth}\n\n"
        "Evaluate how well the generated answer matches the correct answer "
        "on three criteria (0-10 scale):\n\n"
        "1. Correctness: Does the generated answer convey the same factual "
        "information as the correct answer?\n"
        "2. Similarity: How semantically similar is the generated answer to "
        "the correct answer?\n"
        "3. Completeness: Does the generated answer cover all key points "
        "from the correct answer?\n\n"
        "Provide your evaluation in this exact format:\n"
        "Correctness: [0-10]\n"
        "Similarity: [0-10]\n"
        "Completeness: [0-10]\n"
        "Overall: [GOOD or BAD]\n"
        "Reasoning: [brief explanation in 1-2 sentences]\n\n"
        "An answer scoring below 6 on Correctness should always be rated BAD."
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
        correctness = 0.0
        similarity = 0.0
        completeness = 0.0
        overall = "BAD"
        reasoning = ""

        for pattern, field_name in [
            (r"Correctness:\s*(\d+(?:\.\d+)?)", "correctness"),
            (r"Similarity:\s*(\d+(?:\.\d+)?)", "similarity"),
            (r"Completeness:\s*(\d+(?:\.\d+)?)", "completeness"),
        ]:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if field_name == "correctness":
                    correctness = val
                elif field_name == "similarity":
                    similarity = val
                elif field_name == "completeness":
                    completeness = val

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

        return correctness, similarity, completeness, overall, reasoning

    async def _evaluate_answer_async(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        semaphore: Semaphore,
    ) -> AnswerScoreResult:
        async with semaphore:
            try:
                prompt = self.PROMPT_TEMPLATE.format(
                    question=question,
                    answer=answer[:MAX_ANSWER_CHARS],
                    ground_truth=ground_truth[:MAX_GROUND_TRUTH_CHARS],
                )

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.model.generate_content(prompt)
                )

                (correctness, similarity, completeness,
                 overall, reasoning) = self._parse_response(response.text)

                return AnswerScoreResult(
                    answer=answer,
                    correctness_score=correctness,
                    similarity_score=similarity,
                    completeness_score=completeness,
                    overall_quality=overall,
                    reasoning=reasoning,
                )

            except Exception as exc:
                logger.warning("Error evaluating answer: %s", exc)
                return AnswerScoreResult(
                    answer=answer,
                    correctness_score=0.0,
                    similarity_score=0.0,
                    completeness_score=0.0,
                    overall_quality="BAD",
                    reasoning=f"Error: {exc}",
                )

    async def score_answers_async(
        self,
        questions: List[str],
        answers: List[str],
        ground_truths: List[str],
    ) -> List[AnswerScoreResult]:
        semaphore = Semaphore(self.max_concurrent)
        tasks = [
            self._evaluate_answer_async(q, a, gt, semaphore)
            for q, a, gt in zip(questions, answers, ground_truths)
        ]
        return list(await asyncio.gather(*tasks))

    def score_answers(
        self,
        questions: List[str],
        answers: List[str],
        ground_truths: List[str],
    ) -> List[AnswerScoreResult]:
        """Score answers against ground truth (synchronous wrapper)."""
        return _run_async(
            self.score_answers_async(questions, answers, ground_truths)
        )

    def score_single(
        self,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> AnswerScoreResult:
        """Score a single answer against ground truth."""
        results = self.score_answers([question], [answer], [ground_truth])
        return results[0]
