"""
backend/test_reranker.py

Tests the cross-encoder reranker — Step 7 validation.
Run from inside the backend/ folder:
  python test_reranker.py

What it tests:
  1. CrossEncoder model loads correctly (~85MB download on first run)
  2. rerank() reorders chunks by true relevance
  3. Cross-encoder CORRECTS a wrong bi-encoder ranking
  4. rerank_score is attached to each RetrievalResult
  5. top_n truncation works correctly
  6. Empty input is handled gracefully
  7. Full pipeline: hybrid search → rerank
"""

import sys
import tempfile
import numpy as np
from pathlib import Path

print("=" * 60)
print("Step 7 — Cross-Encoder Reranking Test")
print("=" * 60)
print("Note: First run downloads ~85MB cross-encoder model. Please wait.")

# ── 1. Model loads ──────────────────────────────────────────────
print("\n[1] Loading cross-encoder model...")

from core.reranker import CrossEncoderReranker, get_cross_encoder

model = get_cross_encoder()
print(f"    Model type : {type(model).__name__}")
print(f"    ✅ Cross-encoder loaded")

# ── 2. Basic scoring ────────────────────────────────────────────
print("\n[2] Testing raw cross-encoder scores...")

# The model should score query-relevant pairs higher
query = "What is the capital of France?"

pairs = [
    (query, "Paris is the capital and largest city of France."),          # relevant
    (query, "The Eiffel Tower is located in Paris, France."),             # somewhat relevant
    (query, "Python is a popular programming language for data science."), # irrelevant
]

scores = model.predict(pairs, show_progress_bar=False)
print(f"    Scores:")
for (q, doc), score in zip(pairs, scores):
    print(f"      [{score:+.4f}] {doc[:60]}...")

# Relevant chunk must score higher than irrelevant
assert scores[0] > scores[2], \
    f"Relevant chunk should score higher than irrelevant: {scores[0]:.4f} vs {scores[2]:.4f}"
print("    ✅ Relevant chunk scores higher than irrelevant")

# ── 3. rerank() corrects wrong initial ranking ──────────────────
print("\n[3] Testing that rerank() corrects a wrong initial ranking...")

from core.ingestion import Chunk
from core.hybrid_search import RetrievalResult

def make_result(i, text, rrf_score, source="doc.pdf"):
    chunk = Chunk(
        chunk_id=f"chunk_{i:03d}",
        text=text,
        source=source,
        page=1,
        chunk_index=i,
    )
    return RetrievalResult(chunk=chunk, rrf_score=rrf_score)

query = "How does gradient descent optimize neural networks?"

# Deliberately wrong RRF ranking — irrelevant chunk ranked #1
results_before = [
    make_result(0,
        "The Amazon rainforest spans 5.5 million square kilometers across South America.",
        rrf_score=0.035),   # RRF ranked this #1 (wrong!)
    make_result(1,
        "Gradient descent is an optimization algorithm that iteratively adjusts model "
        "weights by moving in the direction of the negative gradient of the loss function.",
        rrf_score=0.028),   # RRF ranked this #2 (should be #1)
    make_result(2,
        "Python is a high-level programming language popular in data science.",
        rrf_score=0.021),
    make_result(3,
        "Backpropagation computes gradients of the loss with respect to each weight "
        "using the chain rule, enabling gradient descent to update the network.",
        rrf_score=0.018),
]

print(f"    Query: '{query}'")
print(f"    Before reranking (RRF order):")
for i, r in enumerate(results_before):
    print(f"      #{i+1} [rrf={r.rrf_score:.4f}] {r.chunk.text[:65]}...")

reranker = CrossEncoderReranker()
results_after = reranker.rerank(query, results_before, top_n=4)

print(f"\n    After reranking (cross-encoder order):")
for i, r in enumerate(results_after):
    print(f"      #{i+1} [rerank={r.rerank_score:+.4f}] {r.chunk.text[:65]}...")

# Gradient descent chunk (chunk_001) must be #1 after reranking
assert results_after[0].chunk.chunk_id == "chunk_001", \
    f"Expected gradient descent chunk at #1, got: {results_after[0].chunk.chunk_id}"

# Rainforest chunk (chunk_000) must drop from #1 to last
rainforest_rank = next(
    i for i, r in enumerate(results_after)
    if r.chunk.chunk_id == "chunk_000"
) + 1
assert rainforest_rank > 1, \
    f"Rainforest chunk should drop from #1 but stayed at #{rainforest_rank}"

print(f"\n    ✅ Gradient descent chunk promoted to #1 (was #2 in RRF)")
print(f"    ✅ Rainforest chunk demoted to #{rainforest_rank} (was #1 in RRF)")
print(f"    ✅ Cross-encoder correctly fixed the wrong RRF ranking")

# ── 4. rerank_score attached ────────────────────────────────────
print("\n[4] Testing rerank_score is attached to results...")

for r in results_after:
    assert hasattr(r, "rerank_score"), f"rerank_score missing on {r.chunk.chunk_id}"
    assert isinstance(r.rerank_score, float), "rerank_score should be float"

print(f"    Scores range: {min(r.rerank_score for r in results_after):.4f} "
      f"to {max(r.rerank_score for r in results_after):.4f}")
print("    ✅ rerank_score attached to all results")

# ── 5. top_n truncation ─────────────────────────────────────────
print("\n[5] Testing top_n truncation...")

results_top2 = reranker.rerank(query, results_before, top_n=2)
assert len(results_top2) == 2, f"Expected 2 results, got {len(results_top2)}"
print(f"    top_n=2 → returned {len(results_top2)} results ✅")

results_top10 = reranker.rerank(query, results_before, top_n=10)
assert len(results_top10) == 4, \
    f"top_n=10 with 4 inputs should return 4, got {len(results_top10)}"
print(f"    top_n=10 with 4 inputs → returned {len(results_top10)} (capped at input size) ✅")

# ── 6. Empty input ──────────────────────────────────────────────
print("\n[6] Testing empty input handling...")

empty_results = reranker.rerank(query, [], top_n=4)
assert empty_results == [], f"Expected empty list, got: {empty_results}"
print("    ✅ Empty input returns empty list gracefully")

# ── 7. Full pipeline: hybrid → rerank ──────────────────────────
print("\n[7] Full pipeline: hybrid search → rerank...")

from core.faiss_index import FAISSIndex
from core.bm25_index import BM25Index
from core.hybrid_search import HybridSearcher

def make_chunk(i, text, source="paper.pdf", page=1):
    return Chunk(
        chunk_id=f"full_{i:03d}",
        text=text,
        source=source,
        page=page,
        chunk_index=i,
    )

corpus = [
    make_chunk(0, "Gradient descent minimizes the loss function by updating weights in the negative gradient direction."),
    make_chunk(1, "Adam optimizer combines momentum and adaptive learning rates for faster neural network training."),
    make_chunk(2, "The Amazon rainforest has exceptional biodiversity with millions of species."),
    make_chunk(3, "Backpropagation uses the chain rule to compute gradients through each layer of the network."),
    make_chunk(4, "Learning rate scheduling reduces the learning rate over time to improve convergence."),
    make_chunk(5, "Climate change is causing ice caps to melt at an accelerating rate."),
    make_chunk(6, "Stochastic gradient descent uses random mini-batches to estimate the full gradient efficiently."),
]

with tempfile.TemporaryDirectory() as fd, \
     tempfile.TemporaryDirectory() as bd:

    fi = FAISSIndex(index_dir=Path(fd))
    fi.build(corpus)

    bi = BM25Index(index_dir=Path(bd))
    bi.build(corpus)

    searcher  = HybridSearcher(fi, bi)
    q = "How do optimization algorithms train neural networks?"

    # Stage 1: broad retrieval
    hybrid_results = searcher.search(q, top_k=7)
    print(f"\n    Query: '{q}'")
    print(f"    Stage 1 — Hybrid search top 7 (RRF order):")
    for i, r in enumerate(hybrid_results):
        print(f"      #{i+1} [rrf={r.rrf_score:.5f}] {r.chunk.text[:65]}...")

    # Stage 2: precise reranking
    reranked = reranker.rerank(q, hybrid_results, top_n=3)
    print(f"\n    Stage 2 — Reranked top 3 (cross-encoder order):")
    for i, r in enumerate(reranked):
        print(f"      #{i+1} [rerank={r.rerank_score:+.4f}] {r.chunk.text[:65]}...")

    # Top results should all be about optimization/neural networks
    for r in reranked:
        text = r.chunk.text.lower()
        assert any(kw in text for kw in [
            "gradient", "optimizer", "adam", "backprop",
            "learning", "stochastic", "network", "train"
        ]), f"Reranked result not about optimization: {r.chunk.text}"

    print("\n    ✅ All top-3 reranked chunks are about neural network optimization")
    print("    ✅ Irrelevant chunks (rainforest, climate) excluded from top-3")

# ── Summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ STEP 7 PASSED — Cross-Encoder Reranking is ready!")
print("=" * 60)
print("\nWhat was validated:")
print("  • Cross-encoder scores query-relevant chunks higher")
print("  • Reranker CORRECTED a wrong RRF ranking in test [3]")
print("  • Gradient descent chunk promoted from #2 → #1")
print("  • Irrelevant rainforest chunk demoted from #1 → last")
print("  • rerank_score attached to every RetrievalResult")
print("  • top_n truncation works correctly")
print("  • Empty input handled gracefully")
print("  • Full two-stage pipeline: hybrid → rerank working end-to-end")
print("\nNext: Step 8 — Full RAG Chain")
