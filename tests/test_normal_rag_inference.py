"""Tests for the dataset-agnostic normal RAG inference experiment."""

from __future__ import annotations

from importlib import import_module
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


def _dataset_module():
    return import_module("normal_rag_inference.dataset")


def _generator_module():
    return import_module("normal_rag_inference.generator")


def _evaluation_module():
    return import_module("normal_rag_inference.evaluation")


def test_parse_merged_context_handles_bulleted_sources() -> None:
    parse_merged_context = _dataset_module().parse_merged_context

    passages = parse_merged_context(
        "- (Doc A) Alpha sentence.\n"
        "- (QA_1) Beta answer.\n"
        "- (Doc A) Alpha sentence.",
        max_chars=100,
    )

    assert [passage.title for passage in passages] == ["Doc A", "QA_1"]
    assert [passage.text for passage in passages] == ["Alpha sentence.", "Beta answer."]


def test_build_qa_corpus_preserves_merged_metadata() -> None:
    build_qa_corpus = _dataset_module().build_qa_corpus

    df = pd.DataFrame(
        [
            {
                "id": "merged-1",
                "question": "what is alpha?",
                "context": "- (Doc A) Alpha context.",
                "answer": "Alpha context.",
                "label": 1,
                "dataset": "asqa",
            },
            {
                "id": "merged-2",
                "question": "what is beta?",
                "context": "- (Doc B) Beta context.",
                "answer": "Beta context.",
                "label": 0,
                "dataset": "wikieval",
            },
        ]
    )

    corpus = build_qa_corpus(df=df, dataset_type="merged", max_passage_chars=200)

    assert [row.row_id for row in corpus.rows] == ["merged-1", "merged-2"]
    assert [row.reference_answer for row in corpus.rows] == ["Alpha context.", "Beta context."]
    assert corpus.rows[0].metadata == {"dataset": "asqa", "label": 1}
    assert corpus.rows[1].metadata == {"dataset": "wikieval", "label": 0}
    assert corpus.documents == ["Alpha context.", "Beta context."]


def test_msmarco_secondary_parser_handles_escaped_newlines() -> None:
    parse_msmarco_context = _dataset_module().parse_msmarco_context

    passages = parse_msmarco_context(
        "[P1] Source: http://example.com/a\\nAlpha text.\\n\\n"
        "[P2] Source: http://example.com/b\\nBeta text.",
        max_chars=100,
    )

    assert [passage.title for passage in passages] == [
        "http://example.com/a",
        "http://example.com/b",
    ]
    assert [passage.text for passage in passages] == ["Alpha text.", "Beta text."]


def test_normal_rag_prompt_never_instructs_abstention() -> None:
    format_rag_prompt = _generator_module().format_rag_prompt

    prompt = format_rag_prompt(
        question="what is arthritis?",
        contexts=["Arthritis is inflammation of the joints."],
        prompt_template=(
            "Answer the question using the retrieved context below. "
            "Give a short, direct answer.\n\n"
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:"
        ),
    )

    assert "[1] Arthritis is inflammation of the joints." in prompt
    assert "I don't know" not in prompt
    assert "cannot answer" not in prompt.lower()
    assert prompt.strip().endswith("Answer:")


def test_parse_plain_answer_removes_chat_markers_and_answer_prefix() -> None:
    parse_plain_answer = _generator_module().parse_plain_answer

    assert parse_plain_answer("<|im_end|> Answer: Alpha text.</s><pad>") == "Alpha text."


def test_causal_instruct_generator_uses_all_retrieved_contexts(monkeypatch) -> None:
    generator_module = _generator_module()
    generator = generator_module.NormalRAGGenerator(
        {
            "model": {
                "backend": "causal_instruct",
                "name": "Qwen/Qwen2.5-7B-Instruct",
                "max_new_tokens": 64,
            },
            "generation": {
                "prompt_template": (
                    "Answer the question using the retrieved context below. "
                    "Give a short, direct answer.\n\n"
                    "Question: {question}\n\n"
                    "Context:\n{context}\n\n"
                    "Answer:"
                )
            },
        }
    )

    def fake_generate(prompts):
        assert len(prompts) == 1
        assert "[1] Alpha context." in prompts[0]
        assert "[2] Beta context." in prompts[0]
        return ["Answer: Alpha context."]

    monkeypatch.setattr(generator, "_generate_causal_instruct_raw", fake_generate)

    result = generator.generate_answer(
        "what is alpha?",
        [
            {"text": "Alpha context.", "score": 0.9},
            {"text": "Beta context.", "score": 0.7},
        ],
    )

    assert result.answer == "Alpha context."
    assert result.best_candidate.retrieval_score == 0.9
    assert len(result.candidates) == 1


def test_output_diagnostics_counts_bad_predictions() -> None:
    compute_output_diagnostics = _evaluation_module().compute_output_diagnostics

    diagnostics = compute_output_diagnostics(
        ["", "I don't know.", "<|im_end|> leaked", "normal answer"]
    )

    assert diagnostics["empty_predictions"] == 1
    assert diagnostics["abstention_predictions"] == 1
    assert diagnostics["special_token_leaks"] == 1
    assert diagnostics["avg_prediction_words"] == 1.75
    assert diagnostics["median_prediction_words"] == 2.0


def test_grouped_metrics_are_reported_for_merged_metadata() -> None:
    compute_grouped_metrics = _evaluation_module().compute_grouped_metrics

    rows = [
        {"predicted_answer": "alpha", "gold_answer": "alpha", "dataset": "asqa", "label": 1},
        {"predicted_answer": "wrong", "gold_answer": "beta", "dataset": "asqa", "label": 0},
        {"predicted_answer": "gamma", "gold_answer": "gamma", "dataset": "wikieval", "label": 1},
    ]

    grouped = compute_grouped_metrics(rows, group_fields=["dataset", "label"])

    assert grouped["dataset"]["asqa"]["n_samples"] == 2
    assert grouped["dataset"]["wikieval"]["exact_match"] == 1.0
    assert grouped["label"]["1"]["n_samples"] == 2
    assert grouped["dataset_label"]["asqa|0"]["exact_match"] == 0.0
