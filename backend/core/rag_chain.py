"""
backend/core/rag_chain.py

WHAT THIS DOES:
  The full RAG pipeline in one place. Takes a raw user query and
  returns a grounded answer with source citations and timing metadata.

  This module is the ONLY thing the FastAPI routes need to call.
  All the complexity (HyDE, hybrid search, reranking) is hidden here.

PIPELINE (in order):
  1. HyDE        — LLM generates hypothetical answer → embed it
  2. HybridSearch — FAISS (dense) + BM25 (sparse) + RRF fusion
                    uses HyDE vector for dense, raw query for sparse
  3. Rerank      — cross-encoder scores top-10 → keep top-4
  4. Build prompt — format context chunks into a structured prompt
  5. LLM answer  — Groq llama-3.1-70b generates a grounded answer
  6. Return       — answer + source chunks + latency breakdown

WHY GROUND THE ANSWER IN CONTEXT?
  Without RAG, the LLM answers from its training data — which may be
  outdated, hallucinated, or wrong for your specific documents.
  By injecting retrieved chunks into the prompt, we force the LLM to
  answer ONLY from what's in your documents. The RAGAS faithfulness
  metric (Step 9) measures how well it stays grounded.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

from core.config import get_settings
from core.ingestion import Chunk
from core.faiss_index import FAISSIndex, get_faiss_index
from core.bm25_index import BM25Index, get_bm25_index
from core.hybrid_search import HybridSearcher, RetrievalResult, get_hybrid_searcher
from core.hyde import HyDEExpander, get_hyde_expander
from core.reranker import CrossEncoderReranker, get_reranker


# ── Prompt templates ──────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the user's question using ONLY the information provided in the context below.

Rules:
- If the context does not contain enough information to answer, say:
  "I cannot find sufficient information in the provided documents to answer this question."
- Do not use any knowledge from outside the provided context.
- Be concise and factual. Cite which source (document name) your answer comes from.
- If multiple chunks support the answer, synthesize them coherently.
- Never fabricate information."""

RAG_USER_TEMPLATE = """Context from retrieved documents:
{context}

Question: {query}

Answer based strictly on the context above:"""


# ── Response data model ───────────────────────────────────────────────────────

@dataclass
class SourceChunk:
    """A source chunk shown to the user alongside the answer."""
    chunk_id: str
    text: str
    source: str
    page: int
    rerank_score: float
    rrf_score: float
    retrieval_source: str      # "dense + sparse" | "dense only" | "sparse only"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "rerank_score": round(self.rerank_score, 4),
            "rrf_score": round(self.rrf_score, 6),
            "retrieval_source": self.retrieval_source,
        }


@dataclass
class RAGResponse:
    """
    Complete response from the RAG pipeline.

    answer          : the LLM-generated answer grounded in context
    source_chunks   : the chunks used to generate the answer
    hypothetical_doc: the HyDE-generated hypothetical answer (for transparency)
    latency         : timing breakdown in seconds for each stage
    metadata        : extra info (model used, chunk counts, etc.)
    """
    answer: str
    source_chunks: List[SourceChunk]
    hypothetical_doc: str
    latency: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "source_chunks": [c.to_dict() for c in self.source_chunks],
            "hypothetical_doc": self.hypothetical_doc,
            "latency": {k: round(v, 3) for k, v in self.latency.items()},
            "metadata": self.metadata,
        }


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(reranked_results: List[RetrievalResult]) -> str:
    """
    Format reranked chunks into a structured context string for the LLM.

    Each chunk is labeled with its source and page number so the LLM
    can cite them accurately. Chunks are ordered by rerank score
    (most relevant first).

    Example output:
        [Source: paper.pdf | Page: 3]
        Gradient descent minimizes the loss function...

        [Source: paper.pdf | Page: 5]
        Backpropagation computes gradients using...
    """
    context_parts = []
    for i, result in enumerate(reranked_results):
        chunk = result.chunk
        part = (
            f"[Source: {chunk.source} | Page: {chunk.page}]\n"
            f"{chunk.text}"
        )
        context_parts.append(part)

    return "\n\n".join(context_parts)


# ── Main RAG chain ────────────────────────────────────────────────────────────

class RAGChain:
    """
    Orchestrates the complete RAG pipeline end-to-end.

    Initialization loads all models once. Query processing is then fast
    (only the LLM call adds significant latency, ~1-3s on Groq).

    Usage:
        chain = RAGChain()
        chain.initialize()   # loads FAISS + BM25 indexes from disk

        response = chain.query("What is attention in transformers?")
        print(response.answer)
        print(response.source_chunks)
    """

    def __init__(self):
        cfg = get_settings()

        # ── LLM for final answer generation ──────────────────────────────────
        self.llm = ChatGroq(
            api_key=cfg.groq_api_key,
            model_name=cfg.groq_model_name,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
        )

        # ── Pipeline components (initialized lazily via initialize()) ─────────
        self.faiss_index: Optional[FAISSIndex] = None
        self.bm25_index: Optional[BM25Index] = None
        self.hybrid_searcher: Optional[HybridSearcher] = None
        self.hyde_expander: Optional[HyDEExpander] = None
        self.reranker: Optional[CrossEncoderReranker] = None

        self._initialized = False
        self.cfg = cfg

    def initialize(self) -> None:
        """
        Load all indexes and models from disk.

        Called once at API startup (FastAPI lifespan event).
        Subsequent queries use cached instances — no reload overhead.

        Raises FileNotFoundError if indexes don't exist yet
        (user hasn't uploaded a PDF). The API handles this gracefully.
        """
        print("[RAGChain] Initializing pipeline components...")

        # Load indexes
        self.faiss_index = get_faiss_index()
        self.bm25_index  = get_bm25_index()

        try:
            self.faiss_index.load()
            self.bm25_index.load()
            print("[RAGChain] Indexes loaded ✅")
        except FileNotFoundError:
            print("[RAGChain] ⚠ No indexes found — upload a PDF first")
            # Don't raise — API should start even without indexes

        # Initialize searcher and models
        self.hybrid_searcher = HybridSearcher(
            self.faiss_index,
            self.bm25_index,
            rrf_k=60,
        )
        self.hyde_expander = None
        self.reranker      = None

        self._initialized = True
        print("[RAGChain] Pipeline ready ✅ (models load on first query)")

    def rebuild_indexes(self, chunks: List[Chunk]) -> None:
        """
        Rebuild FAISS and BM25 indexes from a new list of chunks.

        Called after a new PDF is uploaded and ingested.
        Replaces existing indexes entirely (no merging).
        """
        print(f"[RAGChain] Rebuilding indexes with {len(chunks)} chunks...")

        self.faiss_index.build(chunks)
        self.faiss_index.save()

        self.bm25_index.build(chunks)
        self.bm25_index.save()

        # Refresh hybrid searcher with updated indexes
        self.hybrid_searcher = HybridSearcher(
            self.faiss_index,
            self.bm25_index,
            rrf_k=60,
        )

        print("[RAGChain] Indexes rebuilt and saved ✅")

    def is_ready(self) -> bool:
        """Returns True if indexes are loaded and ready for queries."""
        return (
            self._initialized
            and self.faiss_index is not None
            and self.faiss_index.is_ready()
            and self.bm25_index is not None
            and self.bm25_index.is_ready()
        )

    def query(
        self,
        query: str,
        top_k_retrieval: Optional[int] = None,
        top_n_rerank: Optional[int] = None,
        filter_source: Optional[str] = None,
        use_hyde: bool = True,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline for a user query.

        Args:
            query           : the user's question
            top_k_retrieval : candidates from hybrid search (default: 10)
            top_n_rerank    : chunks to keep after reranking (default: 4)
            filter_source   : restrict retrieval to a specific PDF
            use_hyde        : whether to use HyDE query expansion
                              set False to skip HyDE (faster, less accurate)

        Returns:
            RAGResponse with answer, sources, hypothetical doc, latency

        Raises:
            RuntimeError if indexes are not ready (no PDF uploaded yet)
        """
        if not self.is_ready():
            raise RuntimeError(
                "RAG pipeline not ready. Please upload a PDF first."
            )

        cfg = self.cfg
        top_k = top_k_retrieval or cfg.retrieval_top_k
        top_n = top_n_rerank    or cfg.rerank_top_n
        latency: Dict[str, float] = {}

        # Lazy-load components on first query
        if self.hyde_expander is None:
            self.hyde_expander = get_hyde_expander()
        if self.reranker is None:
            self.reranker = get_reranker()
          
        # ── Stage 1: HyDE query expansion ────────────────────────────────────
        t0 = time.time()

        if use_hyde:
            hypo_text, hyde_vector = self.hyde_expander.expand_with_fallback(
                query, use_cache=True
            )
        else:
            # Skip HyDE — use raw query embedding
            from core.embeddings import embed_query
            hypo_text  = query
            hyde_vector = embed_query(query)

        latency["hyde_seconds"] = round(time.time() - t0, 3)

        # ── Stage 2: Hybrid retrieval ─────────────────────────────────────────
        t1 = time.time()

        hybrid_results = self.hybrid_searcher.search_by_vector(
            query_vector=hyde_vector,
            raw_query=query,
            top_k=top_k,
            faiss_candidates=top_k * 2,
            bm25_candidates=top_k * 2,
            filter_source=filter_source,
        )

        latency["retrieval_seconds"] = round(time.time() - t1, 3)
        print(f"[RAGChain] Retrieved {len(hybrid_results)} candidates "
              f"in {latency['retrieval_seconds']}s")

        # ── Stage 3: Cross-encoder reranking ──────────────────────────────────
        t2 = time.time()

        reranked = self.reranker.rerank(
            query=query,
            results=hybrid_results,
            top_n=top_n,
        )

        latency["rerank_seconds"] = round(time.time() - t2, 3)
        print(f"[RAGChain] Reranked to top {len(reranked)} chunks "
              f"in {latency['rerank_seconds']}s")

        # ── Stage 4: Build context ────────────────────────────────────────────
        context = build_context(reranked)

        # ── Stage 5: LLM answer generation ───────────────────────────────────
        t3 = time.time()

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=RAG_USER_TEMPLATE.format(
                context=context,
                query=query,
            )),
        ]

        try:
            response = self.llm.invoke(messages)
            answer = response.content.strip()
        except Exception as e:
            answer = f"Error generating answer: {e}"

        latency["llm_seconds"] = round(time.time() - t3, 3)
        latency["total_seconds"] = round(
            latency["hyde_seconds"]
            + latency["retrieval_seconds"]
            + latency["rerank_seconds"]
            + latency["llm_seconds"],
            3,
        )

        print(f"[RAGChain] LLM answered in {latency['llm_seconds']}s | "
              f"Total: {latency['total_seconds']}s")

        # ── Stage 6: Build response ───────────────────────────────────────────
        source_chunks = [
            SourceChunk(
                chunk_id=r.chunk.chunk_id,
                text=r.chunk.text,
                source=r.chunk.source,
                page=r.chunk.page,
                rerank_score=getattr(r, "rerank_score", 0.0),
                rrf_score=r.rrf_score,
                retrieval_source=r.source_label(),
            )
            for r in reranked
        ]

        return RAGResponse(
            answer=answer,
            source_chunks=source_chunks,
            hypothetical_doc=hypo_text,
            latency=latency,
            metadata={
                "model": cfg.groq_model_name,
                "query": query,
                "chunks_retrieved": len(hybrid_results),
                "chunks_after_rerank": len(reranked),
                "hyde_used": use_hyde,
                "filter_source": filter_source,
            },
        )


# ── Module-level singleton ─────────────────────────────────────────────────────

_rag_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    """
    Returns the global RAGChain singleton.
    Call initialize() on it before using.
    """
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain
