"""
backend/main.py

WHAT THIS DOES:
  The FastAPI application that exposes the RAG pipeline as a REST API.
  Three main endpoints:
    POST /upload  — ingest a PDF and build search indexes
    POST /query   — run the full RAG pipeline, return answer + RAGAS scores
    GET  /metrics — return aggregated evaluation metrics for the dashboard
    GET  /health  — liveness check for Render deployment monitoring

DESIGN DECISIONS:
  1. Background tasks for RAGAS evaluation:
     RAGAS takes 20-40 seconds. If we block the /query response until
     RAGAS finishes, the user waits too long. Instead we:
       a) Return the RAG answer immediately (~5s)
       b) Run RAGAS in a FastAPI BackgroundTask (non-blocking)
       c) Streamlit polls /metrics to show scores when ready

  2. Lifespan context manager (not @app.on_event):
     FastAPI deprecated on_event in favor of lifespan. We use lifespan
     to load indexes and models once at startup — not on every request.

  3. Single global RAGChain:
     All models (embeddings, cross-encoder, LLM) are loaded once and
     reused across requests. Loading them per-request would take 10s+.

  4. Pydantic request/response models:
     Every endpoint has typed request and response models. This gives
     us automatic validation, OpenAPI docs, and clear contracts between
     frontend and backend.
"""

import json
import shutil
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import get_settings
from core.ingestion import SemanticChunker, ingest_pdf
from core.faiss_index import FAISSIndex, get_faiss_index
from core.bm25_index import BM25Index, get_bm25_index
from core.rag_chain import RAGChain, get_rag_chain
from core.evaluator import RAGASEvaluator, get_evaluator


# ── Lifespan: startup + shutdown ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    print("\n=== Advanced RAG Pipeline starting ===")

    # Create all data directories — works on both /data (Render disk) and local
    try:
        for d in [cfg.data_dir, cfg.faiss_index_path, cfg.bm25_index_path,
                  cfg.uploaded_pdfs_path, cfg.metrics_path]:
            d.mkdir(parents=True, exist_ok=True)
        print(f"[Startup] Data dir ready: {cfg.data_dir}")
    except PermissionError as e:
        print(f"[Startup] WARNING: Cannot write to {cfg.data_dir}: {e}")
        print("[Startup] Falling back to /tmp/rag_data")
        import os
        os.environ["DATA_DIR"] = "/tmp/rag_data"
        # Reload settings with new DATA_DIR
        from core.config import get_settings as _gs
        _gs.cache_clear()
        cfg = _gs()
        for d in [cfg.data_dir, cfg.faiss_index_path, cfg.bm25_index_path,
                  cfg.uploaded_pdfs_path, cfg.metrics_path]:
            d.mkdir(parents=True, exist_ok=True)
        print(f"[Startup] Fallback dir ready: {cfg.data_dir}")

    app.state.chunker = None   # loaded on first upload
    print("=== Startup complete — models load on first request ===\n")
    yield
    print("\nShutting down...")

# ── App initialization ────────────────────────────────────────────────────────

app = FastAPI(
    title="Advanced RAG Pipeline",
    description=(
        "End-to-end RAG system with Hybrid Search (FAISS + BM25 + RRF), "
        "HyDE query expansion, Cross-Encoder reranking, and RAGAS evaluation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit frontend to call this API
# In production, replace "*" with your Streamlit Cloud URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for POST /query"""
    query: str = Field(..., min_length=3, max_length=1000,
                       description="The user's question")
    use_hyde: bool = Field(True,
                           description="Enable HyDE query expansion (recommended)")
    top_k_retrieval: int = Field(10, ge=1, le=50,
                                 description="Candidates to retrieve before reranking")
    top_n_rerank: int = Field(4, ge=1, le=20,
                              description="Chunks to keep after reranking")
    filter_source: Optional[str] = Field(None,
                                         description="Restrict to a specific PDF filename")
    run_evaluation: bool = Field(True,
                                 description="Run RAGAS evaluation in background")


class SourceChunkResponse(BaseModel):
    chunk_id: str
    text: str
    source: str
    page: int
    rerank_score: float
    rrf_score: float
    retrieval_source: str


class QueryResponse(BaseModel):
    """Response from POST /query"""
    answer: str
    source_chunks: List[SourceChunkResponse]
    hypothetical_doc: str
    latency: Dict[str, float]
    metadata: Dict[str, Any]
    evaluation_status: str   # "running", "disabled", "completed"
    eval_id: Optional[str]   # ID to look up scores later in /metrics


class UploadResponse(BaseModel):
    """Response from POST /upload"""
    filename: str
    chunks_created: int
    pages_processed: int
    message: str


class MetricsResponse(BaseModel):
    """Response from GET /metrics"""
    total_queries: int
    avg_rag_latency_secs: float
    metrics: Dict[str, Any]
    recent_evaluations: List[Dict[str, Any]]


# ── Background task: RAGAS evaluation ────────────────────────────────────────

def run_ragas_evaluation(
    query: str,
    rag_response_dict: dict,
    eval_id_placeholder: str,
) -> None:
    """
    Runs RAGAS evaluation in the background after /query returns.

    This function is passed to FastAPI's BackgroundTasks — it runs
    in a thread pool after the HTTP response is sent to the client.

    WHY A DICT INSTEAD OF RAGResponse OBJECT?
      BackgroundTasks run in a separate thread. Passing the full
      RAGResponse object is fine, but converting to dict first ensures
      there are no cross-thread Pydantic serialization issues.
    """
    try:
        from core.rag_chain import RAGResponse, SourceChunk
        from core.evaluator import get_evaluator

        # Reconstruct RAGResponse from dict
        source_chunks = [
            SourceChunk(**chunk) for chunk in rag_response_dict["source_chunks"]
        ]
        rag_response = RAGResponse(
            answer=rag_response_dict["answer"],
            source_chunks=source_chunks,
            hypothetical_doc=rag_response_dict["hypothetical_doc"],
            latency=rag_response_dict["latency"],
            metadata=rag_response_dict["metadata"],
        )

        evaluator = get_evaluator()
        result = evaluator.evaluate(query=query, rag_response=rag_response)
        print(f"[Background RAGAS] eval_id={result.eval_id} "
              f"avg_score={result.average_score()}")

    except Exception as e:
        print(f"[Background RAGAS] ⚠ Evaluation failed: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """
    Liveness endpoint for Render deployment.
    Render pings this every 30 seconds. If it returns non-200,
    Render restarts the service.
    """
    chain = get_rag_chain()
    return {
        "status": "ok",
        "indexes_ready": chain.is_ready(),
        "version": "1.0.0",
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, ingest it, and rebuild the search indexes.

    Steps:
      1. Save the uploaded file to data/uploaded_pdfs/
      2. Run the ingestion pipeline (semantic chunking + metadata)
      3. Rebuild FAISS index from all chunks
      4. Rebuild BM25 index from all chunks
      5. Save both indexes to disk
      6. Return chunk count

    WHY REBUILD INSTEAD OF APPEND?
      For simplicity and correctness. BM25 doesn't support incremental
      updates without rebuilding (rank_bm25 limitation). FAISS does
      support incremental adds, but IDF scores change as corpus grows.
      Rebuilding ensures consistency between both indexes.

    NOTE ON MULTIPLE PDFs:
      We ingest ALL PDFs in uploaded_pdfs/ on each upload — so each
      new upload adds to the existing corpus rather than replacing it.
      To start fresh, delete data/uploaded_pdfs/ and re-upload.
    """
    cfg = get_settings()

    # ── Validate file type ────────────────────────────────────────────────────
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    # ── Save uploaded file ────────────────────────────────────────────────────
    upload_dir = cfg.uploaded_pdfs_path
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    print(f"[Upload] Saved: {file_path} ({len(content) / 1024:.1f} KB)")
    # ── Lazy-load chunker on first upload ─────────────────────────────────────
    if app.state.chunker is None:
        print("[Upload] Loading SemanticChunker for first time...")
        app.state.chunker = SemanticChunker()
    # ── Ingest ALL PDFs in upload directory ───────────────────────────────────
    chunker = app.state.chunker
    all_chunks = []
    all_pdf_paths = list(upload_dir.glob("*.pdf"))

    print(f"[Upload] Ingesting {len(all_pdf_paths)} PDF(s)...")

    for pdf_path in all_pdf_paths:
        try:
            chunks = ingest_pdf(
                pdf_path=pdf_path,
                chunker=chunker,
                save_dir=cfg.data_dir / "chunk_cache",
            )
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[Upload] ⚠ Failed to ingest {pdf_path.name}: {e}")

    if not all_chunks:
        raise HTTPException(
            status_code=422,
            detail=(
                "No text could be extracted from the uploaded PDF. "
                "The file may be scanned/image-based (requires OCR) or corrupt."
            ),
        )

    # ── Rebuild indexes ───────────────────────────────────────────────────────
    chain = get_rag_chain()
    chain.rebuild_indexes(all_chunks)

    # Count pages from the newly uploaded file only
    new_file_chunks = [c for c in all_chunks if c.source == file.filename]
    pages = sorted(set(c.page for c in new_file_chunks))

    print(f"[Upload] Done: {len(all_chunks)} total chunks, "
          f"{len(new_file_chunks)} from {file.filename}")

    return UploadResponse(
        filename=file.filename,
        chunks_created=len(new_file_chunks),
        pages_processed=len(pages),
        message=(
            f"Successfully indexed {file.filename}. "
            f"Total corpus: {len(all_chunks)} chunks from "
            f"{len(all_pdf_paths)} PDF(s). Ready to query."
        ),
    )


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Run the full RAG pipeline for a user question.

    Pipeline:
      query → HyDE → hybrid search (FAISS + BM25 + RRF) →
      cross-encoder rerank → LLM answer → [background: RAGAS eval]

    The RAGAS evaluation runs in the background (non-blocking).
    The answer is returned immediately. Scores appear in /metrics
    a few seconds later once the background task completes.

    Returns:
      answer           : the LLM-generated grounded answer
      source_chunks    : the chunks used, with scores and metadata
      hypothetical_doc : the HyDE-generated hypothetical answer
      latency          : per-stage timing breakdown
      evaluation_status: "running" if RAGAS is in background
      eval_id          : use this to find the evaluation in /metrics
    """
    chain = get_rag_chain()

    if not chain.is_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "No documents indexed yet. "
                "Please upload a PDF via POST /upload first."
            ),
        )

    # ── Run RAG pipeline ──────────────────────────────────────────────────────
    try:
        rag_response = chain.query(
            query=request.query,
            top_k_retrieval=request.top_k_retrieval,
            top_n_rerank=request.top_n_rerank,
            filter_source=request.filter_source,
            use_hyde=request.use_hyde,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {str(e)}",
        )

    # ── Schedule background RAGAS evaluation ──────────────────────────────────
    eval_status = "disabled"
    eval_id = None

    if request.run_evaluation and rag_response.source_chunks:
        response_dict = rag_response.to_dict()
        eval_id = f"pending_{request.query[:20].replace(' ', '_')}"
        eval_status = "running"

        background_tasks.add_task(
            run_ragas_evaluation,
            query=request.query,
            rag_response_dict=response_dict,
            eval_id_placeholder=eval_id,
        )

    # ── Build response ────────────────────────────────────────────────────────
    source_chunks_response = [
        SourceChunkResponse(
            chunk_id=src.chunk_id,
            text=src.text,
            source=src.source,
            page=src.page,
            rerank_score=src.rerank_score,
            rrf_score=src.rrf_score,
            retrieval_source=src.retrieval_source,
        )
        for src in rag_response.source_chunks
    ]

    return QueryResponse(
        answer=rag_response.answer,
        source_chunks=source_chunks_response,
        hypothetical_doc=rag_response.hypothetical_doc,
        latency=rag_response.latency,
        metadata=rag_response.metadata,
        evaluation_status=eval_status,
        eval_id=eval_id,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Return aggregated RAGAS evaluation metrics across all queries.

    Used by the Streamlit dashboard to:
      - Show gauge charts for each metric
      - Plot score history over time
      - Display recent evaluations table

    Returns:
      total_queries        : total queries evaluated so far
      avg_rag_latency_secs : mean end-to-end latency
      metrics              : per-metric mean/min/max/count
      recent_evaluations   : last 20 evaluations with all scores
    """
    evaluator = get_evaluator()
    aggregated = evaluator.get_aggregated_metrics()

    # Load recent evaluations for the history chart
    recent = evaluator.load_history(last_n=20)
    recent_dicts = []
    for r in reversed(recent):   # most recent first
        recent_dicts.append({
            "eval_id"           : r.eval_id,
            "timestamp"         : r.timestamp,
            "query"             : r.query[:80],   # truncate for display
            "faithfulness"      : r.faithfulness,
            "answer_relevancy"  : r.answer_relevancy,
            "context_precision" : r.context_precision,
            "context_recall"    : r.context_recall,
            "average_score"     : r.average_score(),
            "latency_seconds"   : r.latency_seconds,
            "source_files"      : r.source_files,
        })

    return MetricsResponse(
        total_queries=aggregated.get("total_queries", 0),
        avg_rag_latency_secs=aggregated.get("avg_rag_latency_secs", 0.0),
        metrics=aggregated.get("metrics", {}),
        recent_evaluations=recent_dicts,
    )


@app.get("/metrics/history")
async def get_metrics_history(last_n: int = 50):
    """
    Return raw evaluation history as a list.
    Used by Streamlit for plotting score trends over time.
    """
    evaluator = get_evaluator()
    history = evaluator.load_history(last_n=last_n)

    return {
        "count": len(history),
        "evaluations": [
            {
                "timestamp"         : r.timestamp,
                "query"             : r.query[:60],
                "faithfulness"      : r.faithfulness,
                "answer_relevancy"  : r.answer_relevancy,
                "context_precision" : r.context_precision,
                "context_recall"    : r.context_recall,
                "average_score"     : r.average_score(),
            }
            for r in history
        ],
    }


@app.delete("/metrics/clear")
async def clear_metrics():
    """Clear all evaluation history. Useful for resetting the dashboard."""
    evaluator = get_evaluator()
    evaluator.clear_history()
    return {"message": "Evaluation history cleared."}


@app.get("/status")
async def get_status():
    """
    Detailed status of all pipeline components.
    Useful for debugging deployment issues.
    """
    chain = get_rag_chain()
    cfg   = get_settings()

    faiss_ready = (
        chain.faiss_index is not None and chain.faiss_index.is_ready()
    )
    bm25_ready = (
        chain.bm25_index is not None and chain.bm25_index.is_ready()
    )

    # Count uploaded PDFs
    pdf_count = 0
    pdf_names = []
    if cfg.uploaded_pdfs_path.exists():
        pdfs = list(cfg.uploaded_pdfs_path.glob("*.pdf"))
        pdf_count = len(pdfs)
        pdf_names = [p.name for p in pdfs]

    # Count total indexed chunks
    chunk_count = 0
    if faiss_ready:
        chunk_count = chain.faiss_index.index.ntotal

    return {
        "pipeline_ready"  : chain.is_ready(),
        "faiss_index"     : {"ready": faiss_ready, "vectors": chunk_count},
        "bm25_index"      : {"ready": bm25_ready,
                             "documents": len(chain.bm25_index.metadata)
                             if bm25_ready else 0},
        "uploaded_pdfs"   : {"count": pdf_count, "files": pdf_names},
        "models"          : {
            "llm"          : cfg.groq_model_name,
            "embeddings"   : cfg.embedding_model_name,
            "reranker"     : cfg.reranker_model_name,
        },
        "settings"        : {
            "retrieval_top_k": cfg.retrieval_top_k,
            "rerank_top_n"   : cfg.rerank_top_n,
        },
    }
