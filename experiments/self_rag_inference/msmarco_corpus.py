"""MS MARCO corpus preparation and FAISS retrieval for Self-RAG inference."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from rag_filtering.config.loader import resolve_path

logger = logging.getLogger(__name__)

_PASSAGE_MARKER = re.compile(r"\[P\d+\]\s*")
_SOURCE_LINE = re.compile(r"^\s*Source:\s*(?P<source>.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Passage:
    """A single passage extracted from an MS MARCO context cell."""

    title: str
    text: str


@dataclass(frozen=True)
class MSMARCORow:
    """A question row with pointers into the global passage corpus."""

    row_id: str
    question: str
    gold_answer: str
    passage_indices: List[int]


@dataclass(frozen=True)
class MSMARCOCorpus:
    """Prepared MS MARCO rows plus deduplicated retrieval documents."""

    rows: List[MSMARCORow]
    documents: List[str]
    titles: List[str]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_context_segments(context: str) -> List[str]:
    parts = [part.strip() for part in _PASSAGE_MARKER.split(context) if part.strip()]
    if parts:
        return parts
    stripped = context.strip()
    return [stripped] if stripped else []


def _parse_segment(segment: str, max_chars: int) -> Optional[Passage]:
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
    """Split an MS MARCO context cell into deduplicated passages."""

    context = context.replace("\\n", "\n")
    passages: List[Passage] = []
    seen_texts: set[str] = set()
    for segment in _split_context_segments(context):
        passage = _parse_segment(segment, max_chars=max_chars)
        if passage is None:
            continue
        dedupe_key = passage.text.casefold()
        if dedupe_key in seen_texts:
            continue
        seen_texts.add(dedupe_key)
        passages.append(passage)
    return passages


def build_msmarco_corpus(
    df: pd.DataFrame,
    max_passage_chars: int,
    id_column: str = "id",
    question_column: str = "question",
    context_column: str = "context",
    gold_answer_column: str = "gold_answer",
) -> MSMARCOCorpus:
    """Build deduplicated global passage corpus and question rows."""

    documents: List[str] = []
    titles: List[str] = []
    rows: List[MSMARCORow] = []
    doc_to_idx: Dict[str, int] = {}

    for _, row in df.iterrows():
        passage_indices: List[int] = []
        passages = parse_msmarco_context(str(row[context_column]), max_chars=max_passage_chars)
        for passage in passages:
            dedupe_key = passage.text.casefold()
            if dedupe_key not in doc_to_idx:
                doc_to_idx[dedupe_key] = len(documents)
                documents.append(passage.text)
                titles.append(passage.title)
            passage_indices.append(doc_to_idx[dedupe_key])

        rows.append(
            MSMARCORow(
                row_id=str(row[id_column]),
                question=str(row[question_column]),
                gold_answer=str(row[gold_answer_column]),
                passage_indices=passage_indices,
            )
        )

    return MSMARCOCorpus(rows=rows, documents=documents, titles=titles)


def load_msmarco_corpus(data_cfg: Dict[str, Any]) -> MSMARCOCorpus:
    """Load and prepare the configured MS MARCO CSV."""

    csv_path = resolve_path(data_cfg["msmarco_csv"])
    df = pd.read_csv(csv_path)
    max_samples = data_cfg.get("max_samples")
    if max_samples:
        df = df.head(int(max_samples))

    return build_msmarco_corpus(
        df=df,
        max_passage_chars=int(data_cfg["max_passage_chars"]),
        id_column=data_cfg.get("id_column", "id"),
        question_column=data_cfg.get("question_column", "question"),
        context_column=data_cfg.get("context_column", "context"),
        gold_answer_column=data_cfg.get("gold_answer_column", "gold_answer"),
    )


class MSMARCORetriever:
    """FAISS retriever over the deduplicated MS MARCO passage corpus."""

    def __init__(self, cfg: Dict[str, Any], corpus: MSMARCOCorpus) -> None:
        self.cfg = cfg
        self.corpus = corpus
        self.indexer = None

    def _make_rag_config(self) -> Any:
        from rag_filtering.rag.config import RAGConfig

        index_dir = resolve_path(self.cfg["index_dir"])
        return RAGConfig(
            encoder_model=self.cfg["encoder_model"],
            top_k=int(self.cfg["top_k"]),
            normalize_embeddings=bool(self.cfg["normalize_embeddings"]),
            index_batch_size=int(self.cfg["index_batch_size"]),
            index_dir=index_dir,
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
            logger.info("Loading existing MS MARCO index from %s", self.index_dir)
            self.indexer.load_index(self.index_dir)
            return

        logger.info("Building MS MARCO index with %d passages", len(self.corpus.documents))
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
