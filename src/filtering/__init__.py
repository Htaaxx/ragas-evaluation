"""
Filtering module for RAG quality improvement.

Provides:
- AnswerQualityClassifier — learned accept/reject filter (no ground truth needed)
- NLIAnswerFilter — zero-shot NLI-based answer filter (no training needed)
- FilterEvaluator / FilterResult — evaluation harness with 6 required metrics
- FilterDecision — structured accept/reject + confidence output
- AnswerFilter — LLM-as-judge answer scoring vs ground truth
- AnswerRewardFilter — generate-then-score pipeline
- AnswerMetricBundle — RAGAS + lexical answer-correctness metrics
"""

from .ragas_filter import RagasFilter
from .data_split import load_and_split
from .ragas import RAGAS
__all__ = [
    # RAGAS-based filter (new complete pipeline)
    "RagasFilter",
    "RAGAS",
]
