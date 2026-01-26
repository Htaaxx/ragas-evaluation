"""
Document indexing module using FAISS.

This module handles creating and managing FAISS vector indices
for efficient similarity search.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..config import RAGConfig


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
        config: RAGConfig
    ):
        """
        Initialize the document indexer.
        
        Args:
            encoder: SentenceTransformer model for encoding
            config: Configuration object
        """
        self.encoder = encoder
        self.config = config
        
        # Index and docstore
        self.index: Optional[faiss.IndexFlatIP] = None
        self.docstore: List[str] = []
        self.doc_titles: List[str] = []
    
    def build_index(
        self,
        documents: List[str],
        titles: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> faiss.IndexFlatIP:
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of document texts
            titles: Optional list of document titles
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            FAISS index
        """
        batch_size = batch_size or self.config.index_batch_size
        
        print(f"Building FAISS index from {len(documents)} documents...")
        
        # Store documents
        self.docstore = list(documents)
        self.doc_titles = titles if titles else [""] * len(documents)
        
        # Encode documents in batches
        embeddings = []
        num_batches = (len(documents) + batch_size - 1) // batch_size
        
        iterator = range(0, len(documents), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", total=num_batches)
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(documents))
            batch = documents[start_idx:end_idx]
            
            # Encode batch
            batch_embs = self.encoder.encode(
                batch,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False
            )
            
            embeddings.append(batch_embs)
        
        # Concatenate all embeddings
        embeddings_matrix = np.vstack(embeddings).astype("float32")
        
        # Normalize if not already done
        if self.config.normalize_embeddings:
            faiss.normalize_L2(embeddings_matrix)
        
        # Create FAISS index
        embedding_dim = embeddings_matrix.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(embeddings_matrix)
        
        print(f"Built FAISS index: {len(self.docstore)} documents, {embedding_dim} dimensions")
        
        return self.index
    
    def save_index(self, save_dir: Optional[Path] = None) -> None:
        """
        Save index and metadata to disk.
        
        Args:
            save_dir: Directory to save index (default: from config)
        """
        if self.index is None:
            raise RuntimeError("No index to save. Build index first.")
        
        save_dir = Path(save_dir or self.config.index_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = save_dir / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata (docstore and titles)
        meta_path = save_dir / "meta.json"
        metadata = {
            "docstore": self.docstore,
            "doc_titles": self.doc_titles,
            "num_documents": len(self.docstore),
            "embedding_dim": self.index.d if self.index else 0
        }
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Saved index to {save_dir}")
        print(f"   - index.faiss: {index_path}")
        print(f"   - meta.json: {meta_path}")
    
    def load_index(self, load_dir: Path) -> faiss.IndexFlatIP:
        """
        Load index and metadata from disk.
        
        Args:
            load_dir: Directory to load index from
            
        Returns:
            Loaded FAISS index
        """
        load_dir = Path(load_dir)
        
        # Load FAISS index
        index_path = load_dir / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        meta_path = load_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        self.docstore = metadata["docstore"]
        self.doc_titles = metadata.get("doc_titles", [""] * len(self.docstore))
        
        print(f"Loaded index from {load_dir}")
        print(f"   - Documents: {len(self.docstore)}")
        print(f"   - Dimensions: {self.index.d}")
        
        return self.index
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Tuple[List[str], List[float], List[int]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            Tuple of (documents, scores, indices)
        """
        if self.index is None:
            raise RuntimeError("No index loaded. Build or load index first.")
        
        top_k = top_k or self.config.top_k
        
        # Encode query
        query_emb = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings
        ).astype("float32")
        
        # Normalize if needed
        if self.config.normalize_embeddings:
            faiss.normalize_L2(query_emb)
        
        # Search
        scores, indices = self.index.search(query_emb, top_k)
        
        # Extract results
        scores = scores[0].tolist()
        indices = indices[0].tolist()
        documents = [
            self.docstore[idx] 
            for idx in indices 
            if 0 <= idx < len(self.docstore)
        ]
        
        return documents, scores, indices
    
    def get_index_info(self) -> dict:
        """
        Get information about the index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {"status": "No index loaded"}
        
        return {
            "status": "Index loaded",
            "num_documents": len(self.docstore),
            "embedding_dim": self.index.d,
            "index_type": type(self.index).__name__,
            "has_titles": bool(self.doc_titles and any(self.doc_titles))
        }
