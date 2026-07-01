"""Dataset loading and FAISS retrieval for normal RAG inference."""

from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from rag_filtering.config.loader import resolve_path

logger = logging.getLogger(__name__)

_MERGED_BULLET = re.compile(r"^\s*[-*]\s*\((?P<title>[^)]+)\)\s*(?P<text>.*)$")
_MSMARCO_PASSAGE_MARKER = re.compile(r"\[P\d+\]\s*")
_SOURCE_LINE = re.compile(r"^\s*Source:\s*(?P<source>.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Passage:
    """A single retrievable passage."""

    title: str
    text: str


@dataclass(frozen=True)
class QARow:
    """One QA example with metadata preserved for reporting."""

    row_id: str
    question: str
    reference_answer: str
    passage_indices: List[int]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class QACorpus:
    """Prepared QA rows plus a deduplicated retrieval corpus."""

    rows: List[QARow]
    documents: List[str]
    titles: List[str]
    dataset_type: str


def _clean_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _dedupe_passages(passages: List[Passage]) -> List[Passage]:
    deduped: List[Passage] = []
    seen_texts: set[str] = set()
    for passage in passages:
        key = passage.text.casefold()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        deduped.append(passage)
    return deduped


def parse_merged_context(context: str, max_chars: int) -> List[Passage]:
    """Parse merged thesis context cells into deduplicated passages."""

    context = str(context).replace("\\n", "\n")
    passages: List[Passage] = []
    current_title = ""
    current_lines: List[str] = []

    def flush_current() -> None:
        if not current_lines:
            return
        text = _clean_text(" ".join(current_lines))[:max_chars].strip()
        if text:
            passages.append(Passage(title=current_title, text=text))

    for raw_line in context.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _MERGED_BULLET.match(line)
        if match:
            flush_current()
            current_title = match.group("title").strip()
            current_lines = [match.group("text").strip()]
        else:
            current_lines.append(line)

    flush_current()
    if not passages:
        text = _clean_text(context)[:max_chars].strip()
        if text:
            passages.append(Passage(title="", text=text))
    return _dedupe_passages(passages)


def _split_msmarco_segments(context: str) -> List[str]:
    parts = [
        part.strip()
        for part in _MSMARCO_PASSAGE_MARKER.split(context)
        if part.strip()
    ]
    if parts:
        return parts
    stripped = context.strip()
    return [stripped] if stripped else []


def _parse_msmarco_segment(segment: str, max_chars: int) -> Optional[Passage]:
    lines = [line.strip() for line in segment.splitlines() if line.strip()]
    if not lines:
        return None

    title = ""
    body_lines = lines
    source_match = _SOURCE_LINE.match(lines[0])
    if source_match:
        title = source_match.group("source").strip()
        body_lines = lines[1:]

    text = _clean_text(" ".join(body_lines))[:max_chars].strip()
    if not text:
        return None
    return Passage(title=title, text=text)


def parse_msmarco_context(context: str, max_chars: int) -> List[Passage]:
    """Parse an MS MARCO context cell into deduplicated passages."""

    context = str(context).replace("\\n", "\n")
    passages: List[Passage] = []
    for segment in _split_msmarco_segments(context):
        passage = _parse_msmarco_segment(segment, max_chars=max_chars)
        if passage is not None:
            passages.append(passage)
    return _dedupe_passages(passages)


def _row_metadata(row: pd.Series, metadata_columns: List[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for column in metadata_columns:
        if column in row.index:
            value = row[column]
            if hasattr(value, "item"):
                value = value.item()
            metadata[column] = value
    return metadata


def build_qa_corpus(
    df: pd.DataFrame,
    dataset_type: str,
    max_passage_chars: int,
    id_column: str = "id",
    question_column: str = "question",
    context_column: str = "context",
    answer_column: str = "answer",
    metadata_columns: Optional[List[str]] = None,
) -> QACorpus:
    """Build rows and a deduplicated passage corpus for a supported dataset."""

    if metadata_columns is None and dataset_type == "merged":
        metadata_columns = ["dataset", "label"]
    metadata_columns = metadata_columns or []
    documents: List[str] = []
    titles: List[str] = []
    rows: List[QARow] = []
    doc_to_idx: Dict[str, int] = {}

    parser = parse_msmarco_context if dataset_type == "msmarco" else parse_merged_context

    for _, row in df.iterrows():
        passage_indices: List[int] = []
        passages = parser(str(row[context_column]), max_chars=max_passage_chars)
        for passage in passages:
            dedupe_key = passage.text.casefold()
            if dedupe_key not in doc_to_idx:
                doc_to_idx[dedupe_key] = len(documents)
                documents.append(passage.text)
                titles.append(passage.title)
            passage_indices.append(doc_to_idx[dedupe_key])

        rows.append(
            QARow(
                row_id=str(row[id_column]),
                question=str(row[question_column]),
                reference_answer=str(row[answer_column]),
                passage_indices=passage_indices,
                metadata=_row_metadata(row, metadata_columns),
            )
        )

    return QACorpus(
        rows=rows,
        documents=documents,
        titles=titles,
        dataset_type=dataset_type,
    )


def load_qa_corpus(data_cfg: Dict[str, Any]) -> QACorpus:
    """Load a configured QA dataset."""

    csv_path = resolve_path(data_cfg["csv"])
    df = pd.read_csv(csv_path)

    filters = data_cfg.get("filters", {}) or {}
    for column, allowed_values in filters.items():
        if allowed_values is None:
            continue
        if not isinstance(allowed_values, list):
            allowed_values = [allowed_values]
        df = df[df[column].isin(allowed_values)]

    max_samples = data_cfg.get("max_samples")
    if max_samples:
        df = df.head(int(max_samples))

    return build_qa_corpus(
        df=df,
        dataset_type=str(data_cfg["dataset_type"]),
        max_passage_chars=int(data_cfg["max_passage_chars"]),
        id_column=data_cfg.get("id_column", "id"),
        question_column=data_cfg.get("question_column", "question"),
        context_column=data_cfg.get("context_column", "context"),
        answer_column=data_cfg.get("answer_column", "answer"),
        metadata_columns=list(data_cfg.get("metadata_columns", [])),
    )


class NormalRAGRetriever:
    """FAISS retriever over a configured QA corpus."""

    def __init__(self, cfg: Dict[str, Any], corpus: QACorpus) -> None:
        self.cfg = cfg
        self.corpus = corpus
        self.indexer = None

    def _make_rag_config(self) -> Any:
        from rag_filtering.rag.config import RAGConfig

        return RAGConfig(
            encoder_model=self.cfg["encoder_model"],
            top_k=int(self.cfg["top_k"]),
            normalize_embeddings=bool(self.cfg["normalize_embeddings"]),
            index_batch_size=int(self.cfg["index_batch_size"]),
            index_dir=resolve_path(self.cfg["index_dir"]),
        )

    def _make_indexer(self) -> Any:
        from sentence_transformers import SentenceTransformer

        from rag_filtering.rag.retrieval.indexer import DocumentIndexer

        rag_config = self._make_rag_config()
        try:
            encoder = SentenceTransformer(self.cfg["encoder_model"], token=False)
        except TypeError:
            encoder = SentenceTransformer(self.cfg["encoder_model"])
        return DocumentIndexer(encoder=encoder, config=rag_config)

    @property
    def index_dir(self) -> Path:
        return resolve_path(self.cfg["index_dir"])

    def build_or_load(self, force_rebuild: bool = False) -> None:
        """Build a FAISS index or load an existing one from disk."""

        self.indexer = self._make_indexer()
        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "meta.json"
        if not force_rebuild and index_path.exists() and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if int(meta.get("num_documents", -1)) == len(self.corpus.documents):
                logger.info("Loading existing normal RAG index from %s", self.index_dir)
                self.indexer.load_index(self.index_dir)
                return
            logger.info(
                "Rebuilding normal RAG index because existing document count %s "
                "does not match current corpus size %d",
                meta.get("num_documents"),
                len(self.corpus.documents),
            )

        logger.info("Building normal RAG index with %d passages", len(self.corpus.documents))
        self.indexer.build_index(
            documents=self.corpus.documents,
            titles=self.corpus.titles,
            batch_size=int(self.cfg["index_batch_size"]),
        )
        self.indexer.save_index(self.index_dir)

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve top passages for a question."""

        if self.indexer is None:
            raise RuntimeError("Index is not loaded. Call build_or_load() first.")

        documents, scores, indices = self.indexer.search(question, top_k=top_k)
        results: List[Dict[str, Any]] = []
        for doc, score, idx in zip(documents, scores, indices):
            title = self.corpus.titles[idx] if 0 <= idx < len(self.corpus.titles) else ""
            results.append(
                {
                    "text": doc,
                    "title": title,
                    "score": float(score),
                    "index": int(idx),
                }
            )
        return results
