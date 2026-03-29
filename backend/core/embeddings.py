"""
backend/core/embeddings.py

WHAT THIS DOES:
  A thin wrapper around SentenceTransformer that:
  - Loads the model once and caches it (expensive to reload)
  - Exposes embed_texts() for batch embedding (chunks at index time)
  - Exposes embed_query() for single query embedding (at retrieval time)
  - Always L2-normalizes vectors so dot product = cosine similarity
    (required for FAISS IndexFlatIP to behave as cosine search)

WHY NORMALIZE?
  FAISS IndexFlatIP computes inner products (dot products). If vectors
  are L2-normalized (unit length), dot product equals cosine similarity.
  This means score of 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite.
  Without normalization, longer texts get artificially higher scores.

WHY A SINGLETON PATTERN?
  SentenceTransformer takes ~2 seconds to load from disk. If every
  function call reloaded it, the API would be unusably slow. The
  module-level _model variable acts as a process-level cache.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import get_settings

# Module-level singleton — loaded once, reused forever
_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """
    Returns the embedding model, loading it on first call.
    All subsequent calls return the cached instance instantly.
    """
    global _model
    if _model is None:
        cfg = get_settings()
        print(f"[Embeddings] Loading model: {cfg.embedding_model_name}")
        _model = SentenceTransformer(cfg.embedding_model_name)
        print("[Embeddings] Model loaded ✅")
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of texts. Used at INDEX TIME for document chunks.

    Args:
        texts      : list of strings to embed
        batch_size : number of texts to process at once
                     64 is safe for CPU with 8GB RAM
                     reduce to 32 if you get memory errors

    Returns:
        np.ndarray of shape (len(texts), 384), dtype float32
        Vectors are L2-normalized (unit length).
    """
    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,   # only show bar for large batches
        normalize_embeddings=True,            # L2 normalize → cosine sim
        convert_to_numpy=True,
    )

    return embeddings.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string. Used at RETRIEVAL TIME.

    Returns:
        np.ndarray of shape (1, 384), dtype float32
        Normalized — ready to pass directly to FAISS search.

    WHY SHAPE (1, 384) NOT (384,)?
        FAISS expects a 2D array for batch queries. Keeping it 2D
        means the caller never has to reshape — consistent interface.
    """
    model = get_embedding_model()

    embedding = model.encode(
        [text],                              # wrap in list → batch of 1
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embedding.astype(np.float32)     # shape: (1, 384)
