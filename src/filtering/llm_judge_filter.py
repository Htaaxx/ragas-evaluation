"""
llm_judge_filter.py

LLM-as-a-Judge inference module for contextual faithfulness filtering.

Task:
    Given question, context, answer:
    predict whether the answer is grounded in / supported by the context.

Output:
    - filter_label: 1 accepted, 0 rejected
    - filter_confidence: confidence score from judge
    - judge_reason: brief reason
    - judge_raw_output: raw JSON/text returned by model

Default model:
    gpt-4o-mini

Expected input schema:
    Required:
        - id
        - question
        - answer
        - context

    Optional:
        - label
        - gold_ans / gold_answer / reference / reference_answer
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

logger = logging.getLogger(__name__)

RAW_REQUIRED_COLS = ["id", "question", "answer", "context"]


from ..helper import _ensure_path, _normalize_col_aliases, parse_context
from ..evaluation.filter_evaluator import FilterEvaluator


class LLMJudgeFilter:
    """
    LLM-as-a-Judge filter for contextual faithfulness.

    This is an inference-only module, similar to RagasFilter:
        data -> LLM judge predictions -> evaluation

    Label convention:
        1 = accepted / answer is supported by context
        0 = rejected / answer contains unsupported claims
    """

    SYSTEM_PROMPT = """You are a strict evaluator for contextual faithfulness in a RAG system.

Your task is to decide whether the ANSWER is fully supported by the provided CONTEXT.

Important rules:
- Judge ONLY based on the CONTEXT.
- Do NOT use outside knowledge.
- The answer can be factually true in the real world but still rejected if it is not supported by the context.
- Accept minor wording differences and harmless paraphrases.
- Reject if the answer contains any important claim that is unsupported, contradicted, or not inferable from the context.
- If the answer is incomplete but all stated claims are supported, accept it.
- If the context itself is messy or duplicated, still judge whether the answer is supported by the available context.

Return ONLY valid JSON with this exact schema:
{
  "label": 1 or 0,
  "confidence": float between 0 and 1,
  "reason": "brief explanation"
}
"""

    USER_PROMPT_TEMPLATE = """QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Is the answer fully supported by the context?
Return only JSON."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        output_dir: Union[str, Path] = "./results/llm_judge_filter",
        id_col: str = "id",
        question_col: str = "question",
        answer_col: str = "answer",
        context_col: str = "context",
        label_col: str = "label",
        max_context_chars: int = 12000,
        temperature: float = 0.0,
        max_retries: int = 3,
        sleep_between_retries: float = 2.0,
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai package is not installed. Run: pip install openai")

        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is missing.")

        self.client = OpenAI(api_key=self.api_key)

        self.output_dir = _ensure_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.id_col = id_col
        self.question_col = question_col
        self.answer_col = answer_col
        self.context_col = context_col
        self.label_col = label_col

        self.max_context_chars = max_context_chars
        self.temperature = temperature
        self.max_retries = max_retries
        self.sleep_between_retries = sleep_between_retries

        self.output_df: Optional[pd.DataFrame] = None

    def load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        data_path = _ensure_path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        suffix = data_path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(data_path)
        elif suffix == ".jsonl":
            df = pd.read_json(data_path, lines=True)
        elif suffix == ".json":
            df = pd.read_json(data_path)
        elif suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported data file type: {suffix}")

        return _normalize_col_aliases(df)

    def save_df(self, df: pd.DataFrame, output_path: Union[str, Path]) -> None:
        output_path = _ensure_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()

        if suffix == ".csv":
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
        elif suffix == ".jsonl":
            df.to_json(output_path, orient="records", lines=True, force_ascii=False)
        elif suffix == ".json":
            df.to_json(output_path, orient="records", force_ascii=False, indent=2)
        elif suffix == ".parquet":
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output type: {suffix}")

    def prepare_data(self, data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, (str, Path)):
            df = self.load_data(data)
        else:
            df = _normalize_col_aliases(data.copy())

        missing = [c for c in RAW_REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Expected at least {RAW_REQUIRED_COLS}. Current columns: {list(df.columns)}"
            )

        return df

    def build_prompt(self, row: pd.Series) -> str:
        contexts = parse_context(row[self.context_col])
        context_text = "\n\n".join(contexts)

        if len(context_text) > self.max_context_chars:
            context_text = context_text[: self.max_context_chars] + "\n...[TRUNCATED]"

        return self.USER_PROMPT_TEMPLATE.format(
            question=str(row[self.question_col]),
            context=context_text,
            answer=str(row[self.answer_col]),
        )

    def judge_one(self, row: pd.Series) -> Dict[str, Any]:
        user_prompt = self.build_prompt(row)

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )

                raw_output = response.choices[0].message.content or ""
                parsed = self.parse_judge_output(raw_output)

                parsed["judge_raw_output"] = raw_output
                parsed["judge_error"] = None

                return parsed

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM judge failed id=%s attempt=%d/%d: %s",
                    row.get(self.id_col, "<no-id>"),
                    attempt,
                    self.max_retries,
                    exc,
                )
                time.sleep(self.sleep_between_retries)

        return {
            "label": 0,
            "confidence": 0.0,
            "reason": f"Judge failed after {self.max_retries} retries: {last_error}",
            "judge_raw_output": None,
            "judge_error": str(last_error),
        }

    @staticmethod
    def parse_judge_output(raw_output: str) -> Dict[str, Any]:
        try:
            data = json.loads(raw_output)
        except Exception:
            match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
            if not match:
                raise ValueError(f"Cannot parse judge output as JSON: {raw_output}")
            data = json.loads(match.group(0))

        label = int(data.get("label", 0))
        label = 1 if label == 1 else 0

        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        reason = str(data.get("reason", "")).strip()

        return {
            "label": label,
            "confidence": confidence,
            "reason": reason,
        }

    def predict(
        self,
        data: Union[str, Path, pd.DataFrame],
        output_path: Optional[Union[str, Path]] = None,
        resume: bool = True,
        save_every: int = 20,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        df = self.prepare_data(data)

        if output_path is not None:
            output_path = _ensure_path(output_path)

        existing_df = None
        done_ids = set()

        if resume and output_path is not None and output_path.exists():
            existing_df = self.load_data(output_path)
            if self.id_col in existing_df.columns and "filter_label" in existing_df.columns:
                existing_df = existing_df.drop_duplicates(subset=[self.id_col], keep="last")
                done_ids = set(existing_df[self.id_col].tolist())
                logger.info("Resume enabled. Found %d completed samples.", len(done_ids))
                print(f"Found {len(done_ids)} completed samples. Resuming from last checkpoint.")
                print(f"last 5 completed ids: {list(done_ids)[-5:]}")

        rows = []

        pending_df = df[~df[self.id_col].isin(done_ids)].copy()
        total_pending = len(pending_df)

        iterator = pending_df.iterrows()
        if show_progress:
            iterator = tqdm(
                iterator,
                total=total_pending,
                desc=f"LLM Judge ({self.model})",
                unit="sample",
            )

        for _, row in iterator:
            judge = self.judge_one(row)

            out_row = row.to_dict()
            out_row.update(
                {
                    "filter_label": int(judge["label"]),
                    "filter_confidence": float(judge["confidence"]),
                    "judge_reason": judge["reason"],
                    "judge_raw_output": judge["judge_raw_output"],
                    "judge_error": judge["judge_error"],
                }
            )

            rows.append(out_row)

            if output_path is not None and save_every > 0 and len(rows) % save_every == 0:
                partial_df = pd.DataFrame(rows)

                combined_df = (
                    pd.concat([existing_df, partial_df], ignore_index=True)
                    if existing_df is not None
                    else partial_df
                )

                combined_df = combined_df.drop_duplicates(subset=[self.id_col], keep="last")
                self.save_df(combined_df, output_path)

                # Update saved state and clear buffer to avoid repeated concat growth.
                existing_df = combined_df
                rows = []

        new_df = pd.DataFrame(rows)

        if existing_df is not None and len(new_df) > 0:
            output_df = pd.concat([existing_df, new_df], ignore_index=True)
            output_df = output_df.drop_duplicates(subset=[self.id_col], keep="last")
        elif existing_df is not None:
            output_df = existing_df.copy()
        else:
            output_df = new_df

        if len(output_df) > 0:
            order = pd.DataFrame({self.id_col: df[self.id_col].tolist(), "_order": range(len(df))})
            output_df = output_df.merge(order, on=self.id_col, how="left")
            output_df = output_df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

        self.output_df = output_df

        if output_path is not None:
            self.save_df(output_df, output_path)

        return output_df

    def evaluate(
        self,
        df: Optional[pd.DataFrame] = None,
        mode: str = "both",
        evaluator: Optional[Any] = None,
        output_prefix: str = "llm_judge",
    ) -> Dict[str, Any]:
        df = df if df is not None else self.output_df

        if df is None:
            raise ValueError("No prediction output found. Run predict() first.")

        filter_evaluator = FilterEvaluator(
            label_col=self.label_col,
            answer_col=self.answer_col,
            context_col=self.context_col,
            output_dir=self.output_dir,
        )

        return filter_evaluator.evaluate(
            df=df,
            mode=mode,
            evaluator=evaluator,
            output_prefix=output_prefix,
        )

    def run(
        self,
        data: Union[str, Path, pd.DataFrame],
        output_path: Optional[Union[str, Path]] = None,
        eval_mode: str = "both",
        evaluator: Optional[Any] = None,
        resume: bool = True,
        save_every: int = 20,
    ) -> Dict[str, Any]:
        output_df = self.predict(
            data=data,
            output_path=output_path,
            resume=resume,
            save_every=save_every,
        )

        eval_result = self.evaluate(
            df=output_df,
            mode=eval_mode,
            evaluator=evaluator,
        )

        return {
            "output_df": output_df,
            "evaluation": eval_result,
        }
