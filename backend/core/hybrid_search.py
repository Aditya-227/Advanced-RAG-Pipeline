"""
backend/core/hybrid_search.py

WHAT THIS DOES:
  Combines FAISS dense retrieval and BM25 sparse retrieval using
  Reciprocal Rank Fusion (RRF) to produce a single unified ranked list.

HOW RRF WORKS (interview explanation):
  Problem: FAISS scores are cosine similarities in [0, 1].
           BM25 scores are raw TF-IDF values with no upper bound.
           You CANNOT simply add them — the scales are incompatible.

  Solution: Throw away the raw scores. Only use RANKS.

  For each chunk, compute:
    RRF_score = 1/(k + rank_in_faiss) + 1/(k + rank_in_bm25)

  Where:
    rank_in_faiss = position in FAISS results (1 = best, 2 = second, ...)
    rank_in_bm25  = position in BM25 results  (1 = best, 2 = second, ...)
    k = 60 (standard constant, prevents top-1 from dominating too much)

  If a chunk only appears in one system, it gets only one term.
  If it appears in both, both terms add up → higher combined score.

  WHY k=60?
    Published in the original RRF paper (Cormack et al., 2009).
    Empirically shown to be robust across many domains. You could tune
    it, but 60 works well out of the box.

  WHY RRF BEATS SCORE NORMALIZATION?
    Normalizing scores requires knowing the min/max of each system,
    which changes with every query. RRF is parameter-free with respect
    to the score distributions — it always just uses ranks.

WHAT THIS MODULE EXPORTS:
  HybridSearcher  — the main class: takes a query, runs both retrievers,
                    fuses with RRF, returns a ranked list of Chunks.
  get_hybrid_searcher() — module-level singleton factory.
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from core.ingestion import Chunk
from core.faiss_index import FAISSIndex
from core.bm25_index import BM25Index


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    A single chunk returned by hybrid search, with full provenance.

    rrf_score    : the fused RRF score (higher = more relevant)
    faiss_rank   : rank in FAISS results (None if not retrieved by FAISS)
    bm25_rank    : rank in BM25 results  (None if not retrieved by BM25)
    faiss_score  : raw cosine similarity from FAISS (for transparency)
    bm25_score   : raw BM25 score (for transparency)

    Keeping rank + raw score lets the dashboard show WHERE each chunk
    came from — useful for debugging and for explaining the system.
    """
    chunk: Chunk
    rrf_score: float
    faiss_rank: Optional[int] = None
    bm25_rank: Optional[int]  = None
    faiss_score: Optional[float] = None
    bm25_score: Optional[float]  = None

    def source_label(self) -> str:
        """Human-readable label showing which systems retrieved this chunk."""
        if self.faiss_rank is not None and self.bm25_rank is not None:
            return "dense + sparse"
        elif self.faiss_rank is not None:
            return "dense only"
        else:
            return "sparse only"


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    faiss_results: List[Tuple[Chunk, float]],
    bm25_results:  List[Tuple[Chunk, float]],
    k: int = 60,
    top_n: int = 10,
) -> List[RetrievalResult]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    Args:
        faiss_results : List of (Chunk, cosine_score) from FAISS, sorted desc
        bm25_results  : List of (Chunk, bm25_score)   from BM25,  sorted desc
        k             : RRF constant (60 is the standard default)
        top_n         : how many results to return after fusion

    Returns:
        List[RetrievalResult] sorted by rrf_score descending

    STEP BY STEP:
      1. Build a dict keyed by chunk_id, value = RetrievalResult
      2. For each FAISS result at rank r (1-indexed):
           rrf_contribution = 1 / (k + r)
           Add to that chunk's rrf_score
           Record its faiss_rank and faiss_score
      3. Same for BM25 results
      4. Sort all chunks by rrf_score descending
      5. Return top_n
    """
    # chunk_id → RetrievalResult (accumulates scores from both systems)
    fused: Dict[str, RetrievalResult] = {}

    # ── Process FAISS results ─────────────────────────────────────────────────
    for rank, (chunk, score) in enumerate(faiss_results, start=1):
        cid = chunk.chunk_id
        rrf_contrib = 1.0 / (k + rank)

        if cid not in fused:
            fused[cid] = RetrievalResult(chunk=chunk, rrf_score=0.0)

        fused[cid].rrf_score  += rrf_contrib
        fused[cid].faiss_rank  = rank
        fused[cid].faiss_score = score

    # ── Process BM25 results ──────────────────────────────────────────────────
    for rank, (chunk, score) in enumerate(bm25_results, start=1):
        cid = chunk.chunk_id
        rrf_contrib = 1.0 / (k + rank)

        if cid not in fused:
            fused[cid] = RetrievalResult(chunk=chunk, rrf_score=0.0)

        fused[cid].rrf_score += rrf_contrib
        fused[cid].bm25_rank  = rank
        fused[cid].bm25_score = score

    # ── Sort by RRF score and return top_n ────────────────────────────────────
    ranked = sorted(fused.values(), key=lambda r: r.rrf_score, reverse=True)
    return ranked[:top_n]


# ── HybridSearcher ────────────────────────────────────────────────────────────

class HybridSearcher:
    """
    Orchestrates FAISS + BM25 retrieval and fuses results with RRF.

    This is the main retrieval interface for the RAG pipeline.
    HyDE (Step 6) plugs in here by replacing the query vector.

    Usage:
        searcher = HybridSearcher(faiss_idx, bm25_idx)

        # Standard hybrid search
        results = searcher.search("What is attention mechanism?", top_k=10)

        # HyDE: search using a pre-computed vector instead of raw query
        results = searcher.search_by_vector(hyde_vector, raw_query, top_k=10)
    """

    def __init__(
        self,
        faiss_index: FAISSIndex,
        bm25_index: BM25Index,
        rrf_k: int = 60,
    ):
        self.faiss_index = faiss_index
        self.bm25_index  = bm25_index
        self.rrf_k       = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 10,
        faiss_candidates: int = 20,
        bm25_candidates:  int = 20,
        filter_source: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Run hybrid search: FAISS + BM25 → RRF → top_k results.

        Args:
            query            : raw user question
            top_k            : final number of results after fusion
            faiss_candidates : how many candidates to pull from FAISS
                               before fusion (more = better recall,
                               slower fusion — 20 is a good default)
            bm25_candidates  : same for BM25
            filter_source    : restrict to a specific PDF filename

        Returns:
            List[RetrievalResult] sorted by rrf_score descending
        """
        # ── Dense retrieval (FAISS) ───────────────────────────────────────────
        faiss_results = self.faiss_index.search(
            query=query,
            top_k=faiss_candidates,
            filter_source=filter_source,
        )

        # ── Sparse retrieval (BM25) ───────────────────────────────────────────
        bm25_results = self.bm25_index.search(
            query=query,
            top_k=bm25_candidates,
            filter_source=filter_source,
        )

        # ── Fuse with RRF ─────────────────────────────────────────────────────
        results = reciprocal_rank_fusion(
            faiss_results=faiss_results,
            bm25_results=bm25_results,
            k=self.rrf_k,
            top_n=top_k,
        )

        return results

    def search_by_vector(
        self,
        query_vector,           # np.ndarray shape (1, 384)
        raw_query: str,         # original query — used for BM25 (BM25 needs text)
        top_k: int = 10,
        faiss_candidates: int = 20,
        bm25_candidates:  int = 20,
        filter_source: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Hybrid search where FAISS uses a pre-computed vector (for HyDE).

        HyDE generates a hypothetical answer, embeds it, and passes
        that vector here instead of the raw question embedding.
        BM25 still uses the raw query text — BM25 has no concept of
        a "vector", it needs actual words to match against.

        This is the key insight: HyDE only improves DENSE retrieval.
        Sparse retrieval is unaffected and keeps using the original query.
        """
        # ── Dense retrieval using HyDE vector ────────────────────────────────
        faiss_results = self.faiss_index.search_by_vector(
            query_vector=query_vector,
            top_k=faiss_candidates,
        )

        # Apply source filter manually if needed
        if filter_source:
            faiss_results = [
                (c, s) for c, s in faiss_results
                if c.source == filter_source
            ]

        # ── Sparse retrieval using original text query ────────────────────────
        bm25_results = self.bm25_index.search(
            query=raw_query,
            top_k=bm25_candidates,
            filter_source=filter_source,
        )

        # ── Fuse ──────────────────────────────────────────────────────────────
        results = reciprocal_rank_fusion(
            faiss_results=faiss_results,
            bm25_results=bm25_results,
            k=self.rrf_k,
            top_n=top_k,
        )

        return results

    def is_ready(self) -> bool:
        """Both indexes must be loaded for hybrid search to work."""
        return self.faiss_index.is_ready() and self.bm25_index.is_ready()


# ── Module-level singleton ─────────────────────────────────────────────────────

_hybrid_searcher: Optional[HybridSearcher] = None


def get_hybrid_searcher(
    faiss_index: Optional[FAISSIndex] = None,
    bm25_index: Optional[BM25Index] = None,
) -> HybridSearcher:
    """
    Returns global HybridSearcher singleton.
    On first call, requires faiss_index and bm25_index to be passed in.
    Subsequent calls return the cached instance.
    """
    global _hybrid_searcher
    if _hybrid_searcher is None:
        if faiss_index is None or bm25_index is None:
            raise RuntimeError(
                "First call to get_hybrid_searcher() must provide "
                "faiss_index and bm25_index arguments."
            )
        _hybrid_searcher = HybridSearcher(faiss_index, bm25_index)
    return _hybrid_searcher
