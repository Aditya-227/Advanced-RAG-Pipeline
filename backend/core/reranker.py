"""
backend/core/reranker.py

WHAT THIS DOES:
  Takes the top-K chunks from hybrid search and reranks them using a
  cross-encoder model. Returns the top-N most relevant chunks to send
  to the LLM for answer generation.

BI-ENCODER vs CROSS-ENCODER (interview explanation):

  Bi-encoder (what FAISS uses):
    - Encodes query and document SEPARATELY
    - query_vec  = encode("What is attention?")      → 384-dim vector
    - doc_vec    = encode("Attention computes...")   → 384-dim vector
    - score      = dot_product(query_vec, doc_vec)
    - FAST: you pre-compute all doc vectors at index time
    - APPROXIMATE: encoding separately loses interaction signals

  Cross-encoder (what we use here):
    - Encodes query and document TOGETHER
    - score = encode("What is attention? [SEP] Attention computes...") → scalar
    - The model sees both texts simultaneously → captures exact interactions
    - "attention" in query directly influences how "Attention" in doc is weighted
    - SLOW: must run inference for every (query, chunk) pair at query time
    - ACCURATE: much better relevance scores than bi-encoder

  SOLUTION — two-stage pipeline:
    Stage 1 (recall):    bi-encoder retrieves top-20 candidates cheaply
    Stage 2 (precision): cross-encoder reranks those 20 to find true top-4

  This is the industry-standard pattern used by Cohere Rerank, 
  Google's two-tower model, and every serious RAG system.

MODEL:
  cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking dataset (8.8M query-passage pairs)
  - MiniLM-L6 architecture: 6 layers, fast inference, ~85MB
  - Outputs a single relevance score (higher = more relevant)
  - NOT normalized to [0,1] — raw logits, can be negative
"""

from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import CrossEncoder

from core.config import get_settings
from core.hybrid_search import RetrievalResult


# ── Singleton model cache ─────────────────────────────────────────────────────

_cross_encoder: Optional[CrossEncoder] = None


def get_cross_encoder() -> CrossEncoder:
    """
    Load and cache the cross-encoder model.
    First call downloads ~85MB and takes ~10 seconds.
    All subsequent calls return the cached model instantly.
    """
    global _cross_encoder
    if _cross_encoder is None:
        cfg = get_settings()
        print(f"[Reranker] Loading cross-encoder: {cfg.reranker_model_name}")
        _cross_encoder = CrossEncoder(
            cfg.reranker_model_name,
            max_length=512,    # max tokens for query + chunk combined
        )
        print("[Reranker] Cross-encoder loaded ✅")
    return _cross_encoder


# ── Reranker ──────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Reranks a list of RetrievalResult objects using a cross-encoder.

    The cross-encoder receives (query, chunk_text) pairs and outputs
    a relevance score for each. We sort by these scores and return
    the top-N results.

    Usage:
        reranker = CrossEncoderReranker()

        # After hybrid search:
        hybrid_results = searcher.search(query, top_k=20)

        # Rerank to top 4:
        reranked = reranker.rerank(query, hybrid_results, top_n=4)

        # reranked[0] is the most relevant chunk — send to LLM
    """

    def __init__(self):
        # Load model at construction time — fail fast if model missing
        self.model = get_cross_encoder()

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_n: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Rerank retrieval results using cross-encoder relevance scores.

        Args:
            query   : the original user query (raw text)
            results : list of RetrievalResult from hybrid search
            top_n   : how many to keep after reranking
                      if None, uses RERANK_TOP_N from settings

        Returns:
            List[RetrievalResult] sorted by cross-encoder score descending,
            truncated to top_n. Each result has rerank_score added.

        PROCESS:
          1. Build (query, chunk_text) pairs for each result
          2. Run cross-encoder on all pairs in one batch
          3. Attach scores back to RetrievalResult objects
          4. Sort by score descending
          5. Return top_n
        """
        if not results:
            return []

        cfg = get_settings()
        top_n = top_n or cfg.rerank_top_n

        # ── Build input pairs ─────────────────────────────────────────────────
        # CrossEncoder expects: List[Tuple[str, str]]
        pairs = [(query, result.chunk.text) for result in results]

        print(f"[Reranker] Scoring {len(pairs)} chunks with cross-encoder...")

        # ── Score all pairs in one batch ──────────────────────────────────────
        # Returns np.ndarray of shape (len(pairs),) with raw logit scores
        # Higher score = more relevant (no fixed range)
        scores = self.model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False,
        )

        # ── Attach scores to results ──────────────────────────────────────────
        # We add rerank_score as a new attribute to the RetrievalResult
        # (dataclass allows this since we're not frozen)
        scored_results = []
        for result, score in zip(results, scores):
            result.rerank_score = float(score)
            scored_results.append(result)

        # ── Sort by cross-encoder score ───────────────────────────────────────
        scored_results.sort(key=lambda r: r.rerank_score, reverse=True)

        # ── Log reranking changes ─────────────────────────────────────────────
        print(f"[Reranker] Top {min(top_n, len(scored_results))} after reranking:")
        for i, r in enumerate(scored_results[:top_n]):
            print(f"  #{i+1} [rerank={r.rerank_score:.4f}] "
                  f"[rrf={r.rrf_score:.5f}] "
                  f"{r.chunk.text[:60]}...")

        return scored_results[:top_n]

    def rerank_raw_chunks(
        self,
        query: str,
        chunks_with_scores: List[Tuple],
        top_n: Optional[int] = None,
    ) -> List[Tuple]:
        """
        Simpler interface: rerank (chunk, score) tuples directly.

        This is used when we want to rerank without the full
        RetrievalResult wrapper — e.g., in testing or simple pipelines.

        Returns: List of (chunk, rerank_score) tuples, sorted desc.
        """
        from core.ingestion import Chunk

        if not chunks_with_scores:
            return []

        cfg = get_settings()
        top_n = top_n or cfg.rerank_top_n

        chunks = [c for c, _ in chunks_with_scores]
        pairs  = [(query, chunk.text) for chunk in chunks]

        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        ranked = sorted(
            zip(chunks, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:top_n]


# ── Module-level singleton ─────────────────────────────────────────────────────

_reranker: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """
    Returns the global CrossEncoderReranker singleton.
    Model is loaded on first call and cached for all subsequent calls.
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
