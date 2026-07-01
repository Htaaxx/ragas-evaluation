"""Normal RAG answer generation backends."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SPECIAL_MARKUP = re.compile(
    r"</?s>|<pad>|<unk>|<\|im_start\|>|<\|im_end\|>",
    flags=re.IGNORECASE,
)
_ANSWER_PREFIX = re.compile(r"^\s*(?:answer|final answer)\s*:\s*", flags=re.IGNORECASE)


@dataclass(frozen=True)
class GenerationCandidate:
    """Generated answer candidate for one question."""

    answer: str
    raw_output: str
    context: str
    retrieval_score: float


@dataclass(frozen=True)
class GenerationResult:
    """Best answer and scored candidates for a question."""

    answer: str
    best_candidate: GenerationCandidate
    candidates: List[GenerationCandidate]


def format_rag_prompt(
    question: str,
    contexts: List[str],
    prompt_template: str,
) -> str:
    """Format a normal RAG prompt with numbered retrieved contexts."""

    numbered_contexts = "\n\n".join(
        f"[{index}] {context.strip()}"
        for index, context in enumerate(contexts, start=1)
        if context.strip()
    )
    return prompt_template.format(question=question, context=numbered_contexts)


def parse_plain_answer(raw_output: str) -> str:
    """Clean model output into a plain answer string."""

    text = _SPECIAL_MARKUP.sub(" ", raw_output)
    text = re.sub(r"\s+", " ", text).strip()
    text = _ANSWER_PREFIX.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class NormalRAGGenerator:
    """Runtime wrapper for normal RAG generation backends."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.model_cfg = cfg["model"]
        self.generation_cfg = cfg["generation"]
        self.backend = str(self.model_cfg.get("backend", "causal_instruct"))
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    def load_model(self) -> None:
        """Load the configured generator backend."""

        if self.backend == "causal_instruct":
            self._load_causal_instruct()
        else:
            raise ValueError(f"Unknown normal RAG backend: {self.backend}")

    def _load_causal_instruct(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "transformers, accelerate, and bitsandbytes are required for "
                "backend='causal_instruct'."
            ) from exc

        logger.info("Loading normal RAG generator: %s", self.model_cfg["name"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg["name"], token=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        use_cuda = torch.cuda.is_available()
        quantization_config = None
        if self.model_cfg.get("load_in_4bit", use_cuda):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg["name"],
            token=False,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            device_map=self.model_cfg.get("device_map", "auto" if use_cuda else None),
            quantization_config=quantization_config,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def _format_chat_prompts(self, prompts: List[str]) -> List[str]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded for causal instruct backend.")

        system_prompt = str(
            self.model_cfg.get(
                "system_prompt",
                "You are a RAG question-answering assistant. "
                "Use the retrieved context to answer every question.",
            )
        )
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompts

        formatted_prompts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            formatted_prompts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return formatted_prompts

    def _generate_causal_instruct_raw(self, prompts: List[str]) -> List[str]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        chat_prompts = self._format_chat_prompts(prompts)
        encoded = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.model_cfg.get("max_input_tokens", 2048)),
        )
        input_device = next(self.model.parameters()).device
        encoded = encoded.to(input_device)
        input_length = encoded["input_ids"].shape[1]

        temperature = float(self.model_cfg.get("temperature", 0.0))
        generate_kwargs = {
            "max_new_tokens": int(self.model_cfg["max_new_tokens"]),
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": float(self.model_cfg.get("repetition_penalty", 1.05)),
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = float(self.model_cfg.get("top_p", 0.9))

        generated = self.model.generate(**encoded, **generate_kwargs)
        generated_only = generated[:, input_length:]
        return self.tokenizer.batch_decode(generated_only, skip_special_tokens=True)

    def generate_answer(
        self,
        question: str,
        retrieved_passages: List[Dict[str, Any]],
    ) -> GenerationResult:
        """Generate one answer using all retrieved passages."""

        contexts = [str(passage["text"]) for passage in retrieved_passages]
        prompt = format_rag_prompt(
            question=question,
            contexts=contexts,
            prompt_template=self.generation_cfg["prompt_template"],
        )
        raw_output = self._generate_causal_instruct_raw([prompt])[0]
        answer = parse_plain_answer(raw_output)
        best_retrieval_score = max(
            (float(passage["score"]) for passage in retrieved_passages),
            default=0.0,
        )
        candidate = GenerationCandidate(
            answer=answer,
            raw_output=raw_output,
            context="\n\n".join(contexts),
            retrieval_score=best_retrieval_score,
        )
        return GenerationResult(answer=answer, best_candidate=candidate, candidates=[candidate])
