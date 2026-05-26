"""
Document indexing module using FAISS.

Handles creating and managing FAISS vector indices for efficient
similarity search.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_filtering.rag.config import RAGConfig

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Manager for FAISS vector index creation and management.

    Features:
    - Batch encoding of documents
    - FAISS IndexFlatIP (inner product) for similarity search
    - Save/load index with metadata
    """

    def __init__(
        self,
        encoder: SentenceTransformer,
        config: RAGConfig,
    ) -> None:
        self.encoder = encoder
        self.config = config
        self.index: Optional[faiss.IndexFlatIP] = None
        self.docstore: List[str] = []
        self.doc_titles: List[str] = []

    def build_index(
        self,
        documents: List[str],
        titles: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> faiss.IndexFlatIP:
        """Build FAISS index from documents."""
        batch_size = batch_size or self.config.index_batch_size

        logger.info("Building FAISS index from %d documents …", len(documents))

        self.docstore = list(documents)
        self.doc_titles = titles if titles else [""] * len(documents)

        embeddings = []
        num_batches = (len(documents) + batch_size - 1) // batch_size

        iterator = range(0, len(documents), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", total=num_batches)

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(documents))
            batch = documents[start_idx:end_idx]
            batch_embs = self.encoder.encode(
                batch,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False,
            )
            embeddings.append(batch_embs)

        embeddings_matrix = np.vstack(embeddings).astype("float32")

        if self.config.normalize_embeddings:
            faiss.normalize_L2(embeddings_matrix)

        embedding_dim = embeddings_matrix.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(embeddings_matrix)

        logger.info(
            "Built FAISS index: %d documents, %d dimensions",
            len(self.docstore), embedding_dim,
        )
        return self.index

    def save_index(self, save_dir: Optional[Path] = None) -> None:
        """Save index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("No index to save. Build index first.")

        save_dir = Path(save_dir or self.config.index_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        index_path = save_dir / "index.faiss"
        faiss.write_index(self.index, str(index_path))

        meta_path = save_dir / "meta.json"
        metadata = {
            "docstore": self.docstore,
            "doc_titles": self.doc_titles,
            "num_documents": len(self.docstore),
            "embedding_dim": self.index.d if self.index else 0,
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)

        logger.info("Saved index to %s", save_dir)

    def load_index(self, load_dir: Path) -> faiss.IndexFlatIP:
        """Load index and metadata from disk."""
        load_dir = Path(load_dir)

        index_path = load_dir / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        self.index = faiss.read_index(str(index_path))

        meta_path = load_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        self.docstore = metadata["docstore"]
        self.doc_titles = metadata.get("doc_titles", [""] * len(self.docstore))

        logger.info(
            "Loaded index from %s  (%d documents, %d dims)",
            load_dir, len(self.docstore), self.index.d,
        )
        return self.index

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Tuple[List[str], List[float], List[int]]:
        """Search for similar documents."""
        if self.index is None:
            raise RuntimeError("No index loaded. Build or load index first.")

        top_k = top_k or self.config.top_k

        query_emb = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        ).astype("float32")

        if self.config.normalize_embeddings:
            faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, top_k)
        scores_list = scores[0].tolist()
        indices_list = indices[0].tolist()
        documents = [
            self.docstore[idx]
            for idx in indices_list
            if 0 <= idx < len(self.docstore)
        ]

        return documents, scores_list, indices_list

    def get_index_info(self) -> dict:
        """Get information about the index."""
        if self.index is None:
            return {"status": "No index loaded"}
        return {
            "status": "Index loaded",
            "num_documents": len(self.docstore),
            "embedding_dim": self.index.d,
            "index_type": type(self.index).__name__,
            "has_titles": bool(self.doc_titles and any(self.doc_titles)),
        }
