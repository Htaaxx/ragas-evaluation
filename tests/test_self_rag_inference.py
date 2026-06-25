"""Tests for the standalone Self-RAG inference experiment."""

from __future__ import annotations

from importlib import import_module
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


def _msmarco_corpus_module():
    return import_module("self_rag_inference.msmarco_corpus")


def _generator_module():
    return import_module("self_rag_inference.self_rag_generator")


def _evaluation_module():
    return import_module("self_rag_inference.evaluation")


def test_parse_msmarco_context_splits_and_deduplicates_passages() -> None:
    parse_msmarco_context = _msmarco_corpus_module().parse_msmarco_context

    context = (
        "[P1] Source: http://example.com/a\n"
        "Alpha passage text.\n\n"
        "[P2] Source: http://example.com/b\n"
        "Beta passage text.\n\n"
        "[P3] Source: http://example.com/a\n"
        "Alpha passage text."
    )

    passages = parse_msmarco_context(context, max_chars=100)

    assert [p.title for p in passages] == ["http://example.com/a", "http://example.com/b"]
    assert [p.text for p in passages] == ["Alpha passage text.", "Beta passage text."]


def test_parse_msmarco_context_handles_escaped_newlines_from_csv() -> None:
    parse_msmarco_context = _msmarco_corpus_module().parse_msmarco_context

    context = (
        "[P1] Source: http://example.com/a\\nAlpha passage text.\\n\\n"
        "[P2] Source: http://example.com/b\\nBeta passage text."
    )

    passages = parse_msmarco_context(context, max_chars=100)

    assert [p.title for p in passages] == ["http://example.com/a", "http://example.com/b"]
    assert [p.text for p in passages] == ["Alpha passage text.", "Beta passage text."]


def test_build_msmarco_corpus_preserves_rows_and_global_passages() -> None:
    build_msmarco_corpus = _msmarco_corpus_module().build_msmarco_corpus

    df = pd.DataFrame(
        [
            {
                "id": "1",
                "question": "what is alpha?",
                "context": "[P1] Source: s1\nAlpha text.\n\n[P2] Source: s2\nBeta text.",
                "gold_answer": "Alpha text.",
            },
            {
                "id": "2",
                "question": "what is beta?",
                "context": "[P1] Source: s2\nBeta text.",
                "gold_answer": "Beta text.",
            },
        ]
    )

    dataset = build_msmarco_corpus(df, max_passage_chars=200)

    assert len(dataset.rows) == 2
    assert dataset.documents == ["Alpha text.", "Beta text."]
    assert dataset.titles == ["s1", "s2"]


def test_reflection_parser_scores_supported_relevant_outputs() -> None:
    parse_self_rag_output = _generator_module().parse_self_rag_output

    parsed = parse_self_rag_output(
        "[Relevant]The answer is alpha.[Fully supported][Utility:5]</s>",
        score_weights={"relevant": 1.0, "fully_supported": 1.5, "utility": 0.5},
    )

    assert parsed.answer == "The answer is alpha."
    assert parsed.is_relevant is True
    assert parsed.is_fully_supported is True
    assert parsed.utility == 5
    assert parsed.score == 5.0


def test_reflection_parser_removes_generated_padding_and_markup() -> None:
    parse_self_rag_output = _generator_module().parse_self_rag_output

    parsed = parse_self_rag_output(
        (
            "<pad>[Relevant] Private sellers are not acting as a business, "
            "so buyers have fewer consumer protections. "
            "[Fully supported][Utility:5]</s><pad><paragraph>ignored</paragraph>"
        ),
        score_weights={"relevant": 1.0, "fully_supported": 1.5, "utility": 0.5},
    )

    assert parsed.answer == (
        "Private sellers are not acting as a business, "
        "so buyers have fewer consumer protections."
    )
    assert "<pad>" not in parsed.answer
    assert "<paragraph>" not in parsed.answer


def test_candidate_selection_uses_reflection_score_then_retrieval_score() -> None:
    generator = _generator_module()
    GenerationCandidate = generator.GenerationCandidate
    select_best_candidate = generator.select_best_candidate

    weak = GenerationCandidate(
        answer="weak",
        raw_output="[Irrelevant]weak[No support][Utility:1]",
        context="c1",
        retrieval_score=0.99,
        reflection_score=0.5,
        is_relevant=False,
        is_fully_supported=False,
        utility=1,
    )
    strong = GenerationCandidate(
        answer="strong",
        raw_output="[Relevant]strong[Fully supported][Utility:5]",
        context="c2",
        retrieval_score=0.1,
        reflection_score=5.0,
        is_relevant=True,
        is_fully_supported=True,
        utility=5,
    )

    assert select_best_candidate([weak, strong]) == strong


def test_answer_metrics_are_normalized_and_token_based() -> None:
    compute_answer_metrics = _evaluation_module().compute_answer_metrics

    metrics = compute_answer_metrics(
        predictions=["The Alpha, text!"],
        references=["alpha text"],
    )

    assert metrics["exact_match"] == 1.0
    assert metrics["token_f1"] == 1.0
    assert metrics["rouge_l"] == 1.0
