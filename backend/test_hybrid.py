"""
backend/test_hybrid.py

Tests Hybrid Search with RRF — Step 5 validation.
Run from inside the backend/ folder:
  python test_hybrid.py

What it tests:
  1. reciprocal_rank_fusion() math is correct
  2. Chunks in both lists get higher scores than single-system chunks
  3. HybridSearcher.search() returns fused results
  4. HybridSearcher.search_by_vector() works (HyDE path)
  5. RetrievalResult.source_label() shows correct provenance
  6. BM25-only chunks are promoted when FAISS misses them
"""

import sys
import tempfile
import numpy as np
from pathlib import Path

print("=" * 60)
print("Step 5 — Hybrid Search + RRF Test")
print("=" * 60)

# ── Build indexes from scratch for testing ─────────────────────
print("\n[Setup] Building FAISS and BM25 indexes...")

from core.ingestion import Chunk
from core.faiss_index import FAISSIndex
from core.bm25_index import BM25Index
from core.hybrid_search import HybridSearcher, reciprocal_rank_fusion, RetrievalResult

def make_chunk(i, text, source="doc.pdf", page=1):
    return Chunk(
        chunk_id=f"chunk_{i:03d}",
        text=text,
        source=source,
        page=page,
        chunk_index=i,
    )

# Corpus with clear semantic + keyword separation
chunks = [
    make_chunk(0, "BERT is a bidirectional transformer model pre-trained on masked language modeling tasks."),
    make_chunk(1, "GPT-4 is an autoregressive large language model developed by OpenAI for text generation."),
    make_chunk(2, "The Amazon rainforest spans nine countries and contains 10% of all species on Earth."),
    make_chunk(3, "Deforestation in tropical forests releases massive amounts of carbon dioxide into the atmosphere."),
    make_chunk(4, "Retrieval augmented generation combines a retrieval system with a generative language model."),
    make_chunk(5, "FAISS is a library for efficient similarity search and clustering of dense vectors."),
    make_chunk(6, "BM25 is a ranking function used in information retrieval based on term frequency."),
    make_chunk(7, "Python is the most popular programming language for machine learning and data science.", source="guide.pdf"),
]

with tempfile.TemporaryDirectory() as faiss_dir, \
     tempfile.TemporaryDirectory() as bm25_dir:

    faiss_idx = FAISSIndex(index_dir=Path(faiss_dir))
    faiss_idx.build(chunks)

    bm25_idx = BM25Index(index_dir=Path(bm25_dir))
    bm25_idx.build(chunks)

    searcher = HybridSearcher(faiss_idx, bm25_idx, rrf_k=60)

    # ── 1. RRF math ─────────────────────────────────────────────
    print("\n[1] Testing RRF math directly...")

    # Simulate: chunk_000 is rank 1 in FAISS, rank 2 in BM25
    #           chunk_001 is rank 1 in BM25 only
    fake_faiss = [
        (chunks[0], 0.95),   # rank 1
        (chunks[4], 0.80),   # rank 2
        (chunks[5], 0.72),   # rank 3
    ]
    fake_bm25 = [
        (chunks[1], 15.2),   # rank 1
        (chunks[0], 12.1),   # rank 2
        (chunks[4], 9.3),    # rank 3
    ]

    results = reciprocal_rank_fusion(fake_faiss, fake_bm25, k=60, top_n=5)

    print("    RRF scores:")
    for r in results:
        expected_faiss = 1/(60 + r.faiss_rank) if r.faiss_rank else 0
        expected_bm25  = 1/(60 + r.bm25_rank)  if r.bm25_rank  else 0
        expected_total = expected_faiss + expected_bm25
        print(f"      chunk_id={r.chunk.chunk_id}  "
              f"faiss_rank={r.faiss_rank}  bm25_rank={r.bm25_rank}  "
              f"rrf={r.rrf_score:.6f}  expected={expected_total:.6f}")
        assert abs(r.rrf_score - expected_total) < 1e-9, "RRF math wrong!"

    # chunk_000 appears in both lists → should be ranked #1
    assert results[0].chunk.chunk_id == "chunk_000", \
        f"chunk_000 (in both lists) should be #1, got {results[0].chunk.chunk_id}"
    print("    chunk_000 (in both lists) ranked #1: ✅")
    print("    RRF math verified: ✅")

    # ── 2. source_label() provenance ────────────────────────────
    print("\n[2] Testing RetrievalResult.source_label()...")

    for r in results:
        label = r.source_label()
        if r.faiss_rank and r.bm25_rank:
            assert label == "dense + sparse", f"Expected 'dense + sparse', got '{label}'"
        elif r.faiss_rank:
            assert label == "dense only"
        else:
            assert label == "sparse only"
        print(f"      {r.chunk.chunk_id}: {label}")
    print("    ✅ source_label() correct")

    # ── 3. Full hybrid search ────────────────────────────────────
    print("\n[3] Testing HybridSearcher.search()...")

    query = "retrieval augmented generation language model"
    results_hybrid = searcher.search(query, top_k=5)

    print(f"    Query: '{query}'")
    print(f"    Top {len(results_hybrid)} results:")
    for r in results_hybrid:
        print(f"      [rrf={r.rrf_score:.5f}] [{r.source_label():14s}] "
              f"{r.chunk.text[:65]}...")

    assert len(results_hybrid) > 0, "No results from hybrid search"
    # RAG chunk (chunk_004) should be near the top
    top_ids = [r.chunk.chunk_id for r in results_hybrid[:3]]
    assert "chunk_004" in top_ids, \
        f"RAG chunk not in top 3 results: {top_ids}"
    print("    RAG chunk in top 3: ✅")
    print("    ✅ Hybrid search working")

    # ── 4. BM25 promotion test ───────────────────────────────────
    print("\n[4] Testing that BM25 promotes exact keyword matches...")

    # "BM25" is an exact word in chunk_006 — BM25 retriever must find it
    # FAISS might not rank it #1 since "BM25" might not be semantically close
    results_kw = searcher.search("BM25 ranking function", top_k=5)
    print(f"    Query: 'BM25 ranking function'")
    for r in results_kw:
        print(f"      [rrf={r.rrf_score:.5f}] [{r.source_label():14s}] "
              f"{r.chunk.text[:65]}...")

    top_ids = [r.chunk.chunk_id for r in results_kw[:3]]
    assert "chunk_006" in top_ids, \
        f"BM25 chunk (chunk_006) should be in top 3: {top_ids}"
    print("    BM25 chunk in top 3: ✅")

    # ── 5. search_by_vector() — HyDE path ───────────────────────
    print("\n[5] Testing search_by_vector() (HyDE path)...")

    from core.embeddings import embed_query

    # Simulate a HyDE vector: embed a hypothetical answer
    hypothetical_answer = (
        "Retrieval augmented generation (RAG) is a technique where a "
        "retrieval system fetches relevant documents and a generative "
        "model uses them to produce grounded answers."
    )
    hyde_vector = embed_query(hypothetical_answer)

    results_hyde = searcher.search_by_vector(
        query_vector=hyde_vector,
        raw_query="How does retrieval augmented generation work?",
        top_k=5,
    )

    print(f"    HyDE vector for hypothetical answer about RAG")
    print(f"    Top {len(results_hyde)} results:")
    for r in results_hyde:
        print(f"      [rrf={r.rrf_score:.5f}] [{r.source_label():14s}] "
              f"{r.chunk.text[:65]}...")

    assert len(results_hyde) > 0, "No results from vector search"
    top_ids = [r.chunk.chunk_id for r in results_hyde[:3]]
    assert "chunk_004" in top_ids, \
        f"RAG chunk should appear in top 3 for HyDE vector search: {top_ids}"
    print("    RAG chunk in top 3 via HyDE vector: ✅")
    print("    ✅ search_by_vector() working")

    # ── 6. is_ready() ───────────────────────────────────────────
    print("\n[6] Testing is_ready()...")
    assert searcher.is_ready() is True

    empty_faiss = FAISSIndex(index_dir=Path(faiss_dir) / "empty")
    empty_searcher = HybridSearcher(empty_faiss, bm25_idx)
    assert empty_searcher.is_ready() is False
    print("    ✅ is_ready() working")

# ── Summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ STEP 5 PASSED — Hybrid Search with RRF is ready!")
print("=" * 60)
print("\nWhat was validated:")
print("  • RRF math: score = 1/(60+rank_faiss) + 1/(60+rank_bm25)")
print("  • Chunks in BOTH lists score higher than single-system chunks")
print("  • Hybrid search returns semantically + lexically relevant results")
print("  • BM25 successfully promotes exact keyword matches")
print("  • search_by_vector() works for HyDE (Step 6)")
print("  • source_label() shows provenance: dense/sparse/both")
print("\nNext: Step 6 — HyDE Query Expansion")
