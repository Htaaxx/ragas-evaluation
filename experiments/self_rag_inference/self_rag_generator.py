"""Self-RAG answer generation with reflection-token scoring."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_CRITIQUE_TOKEN = re.compile(r"\[(?:Relevant|Irrelevant|Fully supported|Partially supported|No support|Utility:\d+)\]")
_UTILITY = re.compile(r"\[Utility:(?P<utility>\d+)\]")


@dataclass(frozen=True)
class ParsedSelfRAGOutput:
    """Parsed answer and reflection metadata from one Self-RAG output."""

    answer: str
    raw_output: str
    is_relevant: bool
    is_fully_supported: bool
    utility: int
    score: float


@dataclass(frozen=True)
class GenerationCandidate:
    """Generated answer candidate for one retrieved passage."""

    answer: str
    raw_output: str
    context: str
    retrieval_score: float
    reflection_score: float
    is_relevant: bool
    is_fully_supported: bool
    utility: int


@dataclass(frozen=True)
class GenerationResult:
    """Best answer and all scored candidates for a question."""

    answer: str
    best_candidate: GenerationCandidate
    candidates: List[GenerationCandidate]


def format_self_rag_prompt(question: str, paragraph: str, prompt_template: str) -> str:
    """Format a retrieved paragraph using the official Self-RAG prompt shape."""

    return prompt_template.format(question=question, paragraph=paragraph)


def _strip_special_tokens(text: str) -> str:
    text = text.replace("</s>", " ")
    text = text.replace("<s>", " ")
    return re.sub(r"\s+", " ", text).strip()


def _clean_answer(raw_output: str) -> str:
    text = _strip_special_tokens(raw_output)
    text = _CRITIQUE_TOKEN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_self_rag_output(
    raw_output: str,
    score_weights: Dict[str, float],
    fallback_score: float = 0.0,
) -> ParsedSelfRAGOutput:
    """Parse answer text and score Self-RAG reflection tokens."""

    text = _strip_special_tokens(raw_output)
    is_relevant = "[Relevant]" in text
    is_fully_supported = "[Fully supported]" in text
    utility_match = _UTILITY.search(text)
    utility = int(utility_match.group("utility")) if utility_match else 0

    if text:
        score = 0.0
        if is_relevant:
            score += float(score_weights.get("relevant", 0.0))
        if is_fully_supported:
            score += float(score_weights.get("fully_supported", 0.0))
        score += utility * float(score_weights.get("utility", 0.0))
    else:
        score = fallback_score

    return ParsedSelfRAGOutput(
        answer=_clean_answer(raw_output),
        raw_output=raw_output,
        is_relevant=is_relevant,
        is_fully_supported=is_fully_supported,
        utility=utility,
        score=score,
    )


def select_best_candidate(candidates: Iterable[GenerationCandidate]) -> GenerationCandidate:
    """Select candidate by reflection score first, then retrieval score."""

    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("At least one generation candidate is required")
    return max(candidate_list, key=lambda item: (item.reflection_score, item.retrieval_score))


class SelfRAGGenerator:
    """Runtime wrapper for official Self-RAG generation backends."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.model_cfg = cfg["model"]
        self.generation_cfg = cfg["generation"]
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.backend = str(self.model_cfg.get("backend", "vllm"))

    def load_model(self) -> None:
        """Load the configured Self-RAG model backend."""

        if self.backend == "vllm":
            self._load_vllm()
        elif self.backend == "hf":
            self._load_hf()
        else:
            raise ValueError(f"Unknown model backend: {self.backend}")

    def _load_vllm(self) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "vLLM is required for backend='vllm'. Install it on Kaggle with "
                "`pip install vllm` or set model.backend='hf'."
            ) from exc

        logger.info("Loading Self-RAG model with vLLM: %s", self.model_cfg["name"])
        self.model = LLM(
            model=self.model_cfg["name"],
            download_dir=self.model_cfg.get("download_dir"),
            dtype=self.model_cfg.get("dtype", "half"),
            tensor_parallel_size=int(self.model_cfg.get("tensor_parallel_size", 1)),
        )

    def _load_hf(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "transformers, accelerate, and bitsandbytes are required for backend='hf'."
            ) from exc

        fallback_cfg = self.cfg.get("hf_fallback", {})
        quantization_config = None
        if fallback_cfg.get("load_in_4bit", True):
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        logger.info("Loading Self-RAG model with transformers: %s", self.model_cfg["name"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg["name"], token=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg["name"],
            token=False,
            torch_dtype=torch.float16,
            device_map=fallback_cfg.get("device_map", "auto"),
            quantization_config=quantization_config,
        )
        self.model.eval()

    def _generate_raw(self, prompts: List[str]) -> List[str]:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        if self.backend == "vllm":
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=float(self.model_cfg.get("temperature", 0.0)),
                top_p=float(self.model_cfg.get("top_p", 1.0)),
                max_tokens=int(self.model_cfg["max_new_tokens"]),
                skip_special_tokens=False,
            )
            outputs = self.model.generate(prompts, sampling_params)
            return [item.outputs[0].text for item in outputs]

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded for HF backend.")

        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        generated = self.model.generate(
            **encoded,
            max_new_tokens=int(self.model_cfg["max_new_tokens"]),
            do_sample=float(self.model_cfg.get("temperature", 0.0)) > 0.0,
            temperature=max(float(self.model_cfg.get("temperature", 0.0)), 1e-6),
            top_p=float(self.model_cfg.get("top_p", 1.0)),
        )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=False)
        return [text[len(prompt) :].strip() if text.startswith(prompt) else text for text, prompt in zip(decoded, prompts)]

    def generate_answer(
        self,
        question: str,
        retrieved_passages: List[Dict[str, Any]],
    ) -> GenerationResult:
        """Generate and score one candidate per retrieved passage."""

        prompts = [
            format_self_rag_prompt(
                question=question,
                paragraph=str(passage["text"]),
                prompt_template=self.generation_cfg["prompt_template"],
            )
            for passage in retrieved_passages
        ]
        raw_outputs = self._generate_raw(prompts)
        candidates: List[GenerationCandidate] = []
        for raw_output, passage in zip(raw_outputs, retrieved_passages):
            parsed = parse_self_rag_output(
                raw_output=raw_output,
                score_weights=self.generation_cfg["score_weights"],
                fallback_score=float(self.generation_cfg.get("fallback_score", 0.0)),
            )
            candidates.append(
                GenerationCandidate(
                    answer=parsed.answer,
                    raw_output=parsed.raw_output,
                    context=str(passage["text"]),
                    retrieval_score=float(passage["score"]),
                    reflection_score=parsed.score,
                    is_relevant=parsed.is_relevant,
                    is_fully_supported=parsed.is_fully_supported,
                    utility=parsed.utility,
                )
            )

        best = select_best_candidate(candidates)
        return GenerationResult(answer=best.answer, best_candidate=best, candidates=candidates)
