"""
backend/core/bm25_index.py

WHAT THIS DOES:
  Builds a BM25 sparse retrieval index from document chunks and provides
  keyword-based search. Persists the index to disk using joblib.

HOW BM25 WORKS (interview explanation):
  BM25 (Best Match 25) is the gold standard for keyword search — it's
  what Google and Elasticsearch used before neural search. For each
  document it computes a score based on:

    score(q, d) = Σ IDF(term) × TF(term, d) × (k1 + 1)
                              ────────────────────────────
                              TF(term, d) + k1 × (1 - b + b × |d|/avgdl)

  Where:
    IDF(term) = log((N - df + 0.5) / (df + 0.5))
      → Rare terms get higher weight (inverse document frequency)
    TF(term, d) = count of term in document d
      → More occurrences = higher score, but with diminishing returns
    |d| / avgdl = document length / average document length
      → Penalizes long documents that match just by having more words
    k1, b = tuning parameters (default k1=1.5, b=0.75)

  KEY INSIGHT: BM25 is LEXICAL (exact word match), not SEMANTIC.
  "car" does NOT match "automobile" in BM25.
  "car" DOES match "automobile" in FAISS (via embeddings).
  → Combining both = hybrid search (Step 5).

FILES ON DISK:
  data/bm25_index/bm25_model.joblib  — serialized BM25Okapi object
  data/bm25_index/corpus.json        — tokenized corpus + chunk metadata
"""

import re
import json
import joblib
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from rank_bm25 import BM25Okapi

from core.config import get_settings
from core.ingestion import Chunk


# ── Constants ─────────────────────────────────────────────────────────────────

BM25_FILENAME    = "bm25_model.joblib"
CORPUS_FILENAME  = "corpus.json"


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """
    Simple but effective tokenizer for BM25.

    Steps:
      1. Lowercase everything
      2. Remove punctuation (keep alphanumeric + spaces)
      3. Split on whitespace
      4. Remove stopwords (short tokens that carry no meaning)
      5. Remove empty tokens

    WHY NOT USE NLTK?
      We keep this dependency-free. For BM25 this simple tokenizer
      works well in practice. NLTK adds ~50MB and requires corpus
      downloads; the marginal quality gain is not worth it here.

    WHY REMOVE STOPWORDS?
      "the", "a", "is", "of" appear in almost every document so their
      IDF (inverse document frequency) is near zero anyway. Removing
      them speeds up BM25 and reduces index size with no quality loss.
    """
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "is", "was", "are",
        "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may",
        "might", "must", "can", "it", "its", "this", "that", "these",
        "those", "i", "we", "you", "he", "she", "they", "my", "our",
        "your", "his", "her", "their", "what", "which", "who", "not",
    }

    # Lowercase + keep only alphanumeric chars and spaces
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Split and filter
    tokens = [
        token for token in text.split()
        if token not in STOPWORDS and len(token) > 1
    ]

    return tokens


# ── BM25 Index ────────────────────────────────────────────────────────────────

class BM25Index:
    """
    Wraps rank_bm25's BM25Okapi with chunk metadata and persistence.

    The BM25 model stores a tokenized corpus internally. Our metadata
    list is parallel to it: metadata[i] = chunk at corpus position i.

    Usage:
        # Build from scratch
        idx = BM25Index()
        idx.build(chunks)
        idx.save()

        # Load existing
        idx = BM25Index()
        idx.load()

        # Search
        results = idx.search("BERT tokenizer", top_k=10)
    """

    def __init__(self, index_dir: Optional[Path] = None):
        cfg = get_settings()
        self.index_dir = Path(index_dir or cfg.bm25_index_path)

        self.model: Optional[BM25Okapi] = None
        self.metadata: List[dict] = []     # parallel to BM25 corpus
        self.tokenized_corpus: List[List[str]] = []

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, chunks: List[Chunk]) -> None:
        """
        Build BM25 index from a list of Chunk objects.

        Steps:
          1. Tokenize every chunk's text
          2. Build BM25Okapi from the tokenized corpus
          3. Store chunk metadata in parallel list

        BM25Okapi is the Okapi BM25 variant — the most widely used,
        with k1=1.5 (term frequency saturation) and b=0.75
        (document length normalization). These are well-tested defaults.
        """
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunk list")

        print(f"[BM25] Building index from {len(chunks)} chunks...")

        # Tokenize all chunks
        self.tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]

        # Build BM25 model
        # BM25Okapi expects: List[List[str]] (list of tokenized documents)
        self.model = BM25Okapi(self.tokenized_corpus)

        # Parallel metadata list
        self.metadata = [chunk.to_dict() for chunk in chunks]

        print(f"[BM25] Index built ✅")
        print(f"       Documents : {len(self.metadata)}")
        avg_tokens = np.mean([len(t) for t in self.tokenized_corpus])
        print(f"       Avg tokens/chunk: {avg_tokens:.1f}")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add more chunks to an existing index.

        NOTE: BM25Okapi does not support incremental updates — we must
        rebuild the entire model. This is fine for our scale. For
        production at scale you'd use Elasticsearch's BM25 which
        supports incremental indexing natively.
        """
        if self.model is None:
            raise RuntimeError("Index not built yet. Call build() first.")

        # Collect existing chunks + new chunks and rebuild
        existing_chunks = [Chunk.from_dict(m) for m in self.metadata]
        all_chunks = existing_chunks + chunks
        self.build(all_chunks)
        print(f"[BM25] Rebuilt index with {len(all_chunks)} total chunks")

    # ── Persist ────────────────────────────────────────────────────────────────

    def save(self) -> None:
        """
        Save BM25 model and metadata to disk.

        joblib is used for the BM25 model because it handles Python
        objects (classes, arrays) better than pickle for ML objects.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Call build() first.")

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Save BM25 model (binary serialization)
        model_path = self.index_dir / BM25_FILENAME
        joblib.dump(
            {
                "model": self.model,
                "tokenized_corpus": self.tokenized_corpus,
            },
            model_path,
        )

        # Save metadata as JSON (human-readable)
        meta_path = self.index_dir / CORPUS_FILENAME
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

        print(f"[BM25] Saved to {self.index_dir}")
        print(f"       Model file : {model_path.stat().st_size / 1024:.1f} KB")

    def load(self) -> None:
        """Load BM25 model and metadata from disk."""
        model_path = self.index_dir / BM25_FILENAME
        meta_path  = self.index_dir / CORPUS_FILENAME

        if not model_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {model_path}. "
                "Upload a PDF first to build the index."
            )

        saved = joblib.load(model_path)
        self.model            = saved["model"]
        self.tokenized_corpus = saved["tokenized_corpus"]

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"[BM25] Loaded index: {len(self.metadata)} documents")

    def is_ready(self) -> bool:
        """Returns True if model is built/loaded and has documents."""
        return self.model is not None and len(self.metadata) > 0

    # ── Search ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_source: Optional[str] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for the most relevant chunks using BM25 keyword scoring.

        Args:
            query         : the user's question (raw text)
            top_k         : number of results to return
            filter_source : if set, only return chunks from this PDF filename

        Returns:
            List of (Chunk, score) tuples, sorted by score descending.
            BM25 scores are NOT in [0, 1] — they're raw values that
            depend on corpus size and term frequencies. This is fine
            because RRF (Step 5) uses RANKS not raw scores.

        HOW IT WORKS:
          1. Tokenize the query the same way we tokenized the corpus
          2. BM25 scores every document against the query
          3. Return top_k by score (with optional source filter)
        """
        if not self.is_ready():
            raise RuntimeError("BM25 index not ready. Build or load it first.")

        # Tokenize query — must use same tokenizer as corpus
        query_tokens = tokenize(query)

        if not query_tokens:
            # All query words were stopwords — return empty
            print("[BM25] Warning: query reduced to empty after tokenization")
            return []

        # Get BM25 scores for all documents
        # scores[i] = BM25 score of document i for this query
        scores = self.model.get_scores(query_tokens)

        # Get sorted indices (highest score first)
        ranked_indices = np.argsort(scores)[::-1]

        results: List[Tuple[Chunk, float]] = []

        for idx in ranked_indices:
            score = float(scores[idx])

            # Skip documents with zero score (no query terms match)
            if score <= 0.0:
                break

            chunk = Chunk.from_dict(self.metadata[idx])

            # Apply source filter if requested
            if filter_source and chunk.source != filter_source:
                continue

            results.append((chunk, score))

            if len(results) >= top_k:
                break

        return results

    def get_all_scores(self, query: str) -> np.ndarray:
        """
        Returns raw BM25 scores for ALL documents.

        Used internally by hybrid search (RRF) when we need the full
        ranking, not just the top-k, to compute reciprocal ranks.
        """
        if not self.is_ready():
            raise RuntimeError("BM25 index not ready.")

        query_tokens = tokenize(query)
        if not query_tokens:
            return np.zeros(len(self.metadata))

        return self.model.get_scores(query_tokens)


# ── Module-level singleton ─────────────────────────────────────────────────────

_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Returns the global BM25Index instance, creating it if needed."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index
