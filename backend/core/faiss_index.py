"""
backend/core/faiss_index.py

WHAT THIS DOES:
  Builds a FAISS dense vector index from document chunks and provides
  similarity search. Everything persists to disk so the index survives
  server restarts.

HOW FAISS WORKS (interview explanation):
  FAISS (Facebook AI Similarity Search) stores vectors in a flat array
  and supports extremely fast nearest-neighbor search. We use
  IndexFlatIP (Inner Product) which computes dot products between the
  query vector and every stored vector. With L2-normalized vectors,
  this equals cosine similarity. "Flat" means no approximation — every
  vector is checked. For our scale (thousands of chunks) this is fast
  enough. For millions of vectors you'd switch to IndexIVFFlat (clusters
  for approximate search) but accuracy would drop slightly.

FILES ON DISK:
  data/faiss_index/index.faiss   — the FAISS binary index (vectors)
  data/faiss_index/metadata.json — chunk metadata (text, source, page)
                                   FAISS stores vectors, not text.
                                   We link them via positional index.
"""

import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from core.config import get_settings
from core.ingestion import Chunk
from core.embeddings import embed_texts, embed_query


# ── Constants ────────────────────────────────────────────────────────────────

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"


# ── Index builder ─────────────────────────────────────────────────────────────

class FAISSIndex:
    """
    Wraps a FAISS flat inner-product index with chunk metadata.

    The FAISS index stores vectors at positions 0, 1, 2, ...
    The metadata list stores chunk dicts at the same positions.
    So FAISS result "position 42" → metadata[42] → the full Chunk.

    Usage:
        # Build from scratch
        idx = FAISSIndex()
        idx.build(chunks)
        idx.save()

        # Load existing
        idx = FAISSIndex()
        idx.load()

        # Search
        results = idx.search("what is machine learning?", top_k=5)
    """

    def __init__(self, index_dir: Optional[Path] = None):
        cfg = get_settings()
        self.index_dir = index_dir or cfg.faiss_index_path
        self.index_dir = Path(self.index_dir)

        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[dict] = []          # parallel list to FAISS index
        self.embedding_dim: int = 384           # all-MiniLM-L6-v2 output dim

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: List[Chunk], batch_size: int = 64) -> None:
        """
        Build the FAISS index from a list of Chunk objects.

        Steps:
          1. Extract text from each chunk
          2. Embed in batches (avoids OOM on large docs)
          3. Create FAISS IndexFlatIP
          4. Add all vectors at once
          5. Store metadata in parallel list

        Args:
            chunks     : list of Chunk objects from ingestion pipeline
            batch_size : embedding batch size (reduce if memory errors)
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        print(f"[FAISS] Building index from {len(chunks)} chunks...")

        # Step 1+2: Embed all chunk texts
        texts = [chunk.text for chunk in chunks]
        embeddings = embed_texts(texts, batch_size=batch_size)
        # embeddings shape: (num_chunks, 384), float32, L2-normalized

        # Step 3: Create index
        # IndexFlatIP = exact inner product search
        # With normalized vectors: inner product = cosine similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Step 4: Add vectors
        # FAISS assigns sequential IDs: 0, 1, 2, ...
        self.index.add(embeddings)

        # Step 5: Store metadata at same positions as FAISS IDs
        self.metadata = [chunk.to_dict() for chunk in chunks]

        print(f"[FAISS] Index built ✅")
        print(f"        Total vectors: {self.index.ntotal}")
        print(f"        Vector dim   : {self.embedding_dim}")

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> None:
        """
        Add more chunks to an existing index (incremental updates).
        Call build() first if the index doesn't exist yet.
        """
        if self.index is None:
            raise RuntimeError("Index not built yet. Call build() first.")

        texts = [chunk.text for chunk in chunks]
        embeddings = embed_texts(texts, batch_size=batch_size)

        self.index.add(embeddings)
        self.metadata.extend([chunk.to_dict() for chunk in chunks])

        print(f"[FAISS] Added {len(chunks)} chunks. Total: {self.index.ntotal}")

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self) -> None:
        """
        Save index and metadata to disk.

        FAISS has its own binary format (.faiss file).
        Metadata is saved as JSON alongside it.
        """
        if self.index is None:
            raise RuntimeError("No index to save. Call build() first.")

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS binary index
        index_path = self.index_dir / INDEX_FILENAME
        faiss.write_index(self.index, str(index_path))

        # Save metadata JSON
        meta_path = self.index_dir / METADATA_FILENAME
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

        print(f"[FAISS] Saved to {self.index_dir}")
        print(f"        Index file : {index_path.stat().st_size / 1024:.1f} KB")
        print(f"        Metadata   : {meta_path.stat().st_size / 1024:.1f} KB")

    def load(self) -> None:
        """
        Load index and metadata from disk.
        Called at API startup so search is immediately available.
        """
        index_path = self.index_dir / INDEX_FILENAME
        meta_path = self.index_dir / METADATA_FILENAME

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Upload a PDF first to build the index."
            )

        self.index = faiss.read_index(str(index_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"[FAISS] Loaded index: {self.index.ntotal} vectors")

    def is_ready(self) -> bool:
        """Returns True if index is built/loaded and has vectors."""
        return self.index is not None and self.index.ntotal > 0

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_source: Optional[str] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for the most relevant chunks for a query.

        Args:
            query         : the user's question (raw text)
            top_k         : number of results to return
            filter_source : if set, only return chunks from this PDF filename

        Returns:
            List of (Chunk, score) tuples, sorted by score descending.
            Score is cosine similarity in [0, 1] (higher = more similar).

        HOW IT WORKS:
          1. Embed the query → 384-dim vector
          2. FAISS computes dot product with every stored vector
          3. Returns indices of top_k highest scores
          4. We look up chunk metadata by those indices
          5. Optionally filter by source filename
        """
        if not self.is_ready():
            raise RuntimeError("Index not ready. Build or load it first.")

        # Embed query — shape (1, 384)
        query_vec = embed_query(query)

        # If filtering by source, we need to retrieve more candidates
        # then filter down, because FAISS doesn't support metadata filters natively
        search_k = top_k * 5 if filter_source else top_k

        # FAISS search: returns (scores, indices) both shape (1, search_k)
        scores, indices = self.index.search(query_vec, min(search_k, self.index.ntotal))

        results: List[Tuple[Chunk, float]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 for "not enough results"
                continue

            chunk_dict = self.metadata[idx]
            chunk = Chunk.from_dict(chunk_dict)

            # Apply metadata filter if requested
            if filter_source and chunk.source != filter_source:
                continue

            results.append((chunk, float(score)))

            if len(results) >= top_k:
                break

        return results

    def search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search using a pre-computed embedding vector.

        This is used by HyDE — which generates a hypothetical document
        embedding BEFORE searching, rather than using the raw query text.

        Args:
            query_vector : shape (1, 384), float32, L2-normalized
            top_k        : number of results to return
        """
        if not self.is_ready():
            raise RuntimeError("Index not ready. Build or load it first.")

        scores, indices = self.index.search(
            query_vector, min(top_k, self.index.ntotal)
        )

        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = Chunk.from_dict(self.metadata[idx])
            results.append((chunk, float(score)))

        return results


# ── Module-level singleton ────────────────────────────────────────────────────
# The FastAPI app creates one FAISSIndex instance and reuses it.
# Import and use this directly in your route handlers.

_faiss_index: Optional[FAISSIndex] = None


def get_faiss_index() -> FAISSIndex:
    """Returns the global FAISSIndex instance, creating it if needed."""
    global _faiss_index
    if _faiss_index is None:
        _faiss_index = FAISSIndex()
    return _faiss_index
