"""Self-RAG answer generation with reflection-token scoring."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_CRITIQUE_TOKEN = re.compile(
    r"\[(?:Relevant|Irrelevant|Fully supported|Partially supported|No support|Utility:\d+)\]"
)
_UTILITY = re.compile(r"\[Utility:(?P<utility>\d+)\]")
_ANSWER_TERMINATOR = re.compile(
    r"\[(?:Fully supported|Partially supported|No support|Utility:\d+)\]"
)
_SPECIAL_MARKUP = re.compile(
    r"</?s>|<pad>|<unk>|<\|im_start\|>|<\|im_end\|>|\[Retrieval\]|</?paragraph>",
    flags=re.IGNORECASE,
)


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


def format_seq2seq_prompt(
    question: str,
    contexts: List[str],
    prompt_template: str,
) -> str:
    """Format a normal RAG prompt using all retrieved contexts."""

    numbered_contexts = "\n\n".join(
        f"[{index}] {context.strip()}"
        for index, context in enumerate(contexts, start=1)
        if context.strip()
    )
    return prompt_template.format(question=question, context=numbered_contexts)


def _strip_special_tokens(text: str) -> str:
    text = _SPECIAL_MARKUP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_answer(raw_output: str) -> str:
    text = _strip_special_tokens(raw_output)
    text = re.sub(r"^\[(?:Relevant|Irrelevant)\]\s*", "", text)
    text = _ANSWER_TERMINATOR.split(text, maxsplit=1)[0]
    text = _CRITIQUE_TOKEN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_plain_answer(raw_output: str) -> str:
    """Clean a plain seq2seq answer without Self-RAG reflection scoring."""

    return _strip_special_tokens(raw_output)


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
        elif self.backend == "seq2seq":
            self._load_seq2seq()
        elif self.backend == "causal_instruct":
            self._load_causal_instruct()
        else:
            raise ValueError(f"Unknown model backend: {self.backend}")

    def _load_vllm(self) -> None:
        try:
            LLM = import_module("vllm").LLM
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
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token or self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg["name"],
            token=False,
            torch_dtype=torch.float16,
            device_map=fallback_cfg.get("device_map", "auto"),
            quantization_config=quantization_config,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.eval()

    def _load_seq2seq(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and accelerate are required for backend='seq2seq'."
            ) from exc

        logger.info("Loading seq2seq RAG generator: %s", self.model_cfg["name"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg["name"], token=False)

        use_cuda = torch.cuda.is_available()
        torch_dtype = torch.float16 if use_cuda else torch.float32
        device_map = self.model_cfg.get("device_map", "auto" if use_cuda else None)
        kwargs = {
            "token": False,
            "torch_dtype": torch_dtype,
        }
        if device_map is not None:
            kwargs["device_map"] = device_map

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_cfg["name"],
            **kwargs,
        )
        if device_map is None:
            self.model.to("cuda" if use_cuda else "cpu")
        self.model.eval()

    def _load_causal_instruct(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "transformers, accelerate, and bitsandbytes are required for "
                "backend='causal_instruct'."
            ) from exc

        logger.info("Loading causal instruct RAG generator: %s", self.model_cfg["name"])
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

    def _generate_raw(self, prompts: List[str]) -> List[str]:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        if self.backend == "vllm":
            SamplingParams = import_module("vllm").SamplingParams

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

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        input_length = encoded["input_ids"].shape[1]
        temperature = float(self.model_cfg.get("temperature", 0.0))
        generate_kwargs = {
            "max_new_tokens": int(self.model_cfg["max_new_tokens"]),
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = float(self.model_cfg.get("top_p", 1.0))

        generated = self.model.generate(**encoded, **generate_kwargs)
        generated_only = generated[:, input_length:]
        return self.tokenizer.batch_decode(generated_only, skip_special_tokens=False)

    def _generate_seq2seq_raw(self, prompts: List[str]) -> List[str]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.model_cfg.get("max_input_tokens", 1024)),
        )
        input_device = next(self.model.parameters()).device
        encoded = encoded.to(input_device)

        temperature = float(self.model_cfg.get("temperature", 0.0))
        generate_kwargs = {
            "max_new_tokens": int(self.model_cfg["max_new_tokens"]),
            "num_beams": int(self.model_cfg.get("num_beams", 4)),
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = float(self.model_cfg.get("top_p", 1.0))

        generated = self.model.generate(**encoded, **generate_kwargs)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def _format_causal_chat_prompts(self, prompts: List[str]) -> List[str]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded for causal instruct backend.")

        system_prompt = str(
            self.model_cfg.get(
                "system_prompt",
                "You are a precise question-answering assistant. "
                "Answer using only the provided context.",
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

        chat_prompts = self._format_causal_chat_prompts(prompts)
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
        """Generate and score one candidate per retrieved passage."""

        if self.backend in {"seq2seq", "causal_instruct"}:
            contexts = [str(passage["text"]) for passage in retrieved_passages]
            prompt = format_seq2seq_prompt(
                question=question,
                contexts=contexts,
                prompt_template=self.generation_cfg["prompt_template"],
            )
            if self.backend == "seq2seq":
                raw_output = self._generate_seq2seq_raw([prompt])[0]
            else:
                raw_output = self._generate_causal_instruct_raw([prompt])[0]
            answer = parse_plain_answer(raw_output)
            best_retrieval_score = max(
                (float(passage["score"]) for passage in retrieved_passages),
                default=0.0,
            )
            joined_context = "\n\n".join(contexts)
            candidate = GenerationCandidate(
                answer=answer,
                raw_output=raw_output,
                context=joined_context,
                retrieval_score=best_retrieval_score,
                reflection_score=best_retrieval_score,
                is_relevant=True,
                is_fully_supported=True,
                utility=0,
            )
            return GenerationResult(
                answer=answer,
                best_candidate=candidate,
                candidates=[candidate],
            )

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
