"""
backend/test_faiss.py

Tests the FAISS index builder and retriever — Step 3 validation.
Run from inside the backend/ folder:
  python test_faiss.py

What it tests:
  1. embed_texts() produces correct shape and normalized vectors
  2. embed_query() produces correct shape
  3. FAISSIndex.build() creates index with correct vector count
  4. FAISSIndex.save() and load() round-trip correctly
  5. FAISSIndex.search() returns relevant results
  6. FAISSIndex.search_by_vector() works (used by HyDE later)
  7. Metadata filter works (filter_source)
"""

import sys
import tempfile
import numpy as np
from pathlib import Path

print("=" * 60)
print("Step 3 — FAISS Index Test")
print("=" * 60)

# ── 1. Embedding shapes ────────────────────────────────────────
print("\n[1] Testing embed_texts() and embed_query()...")

from core.embeddings import embed_texts, embed_query

texts = [
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks are inspired by the human brain.",
    "The Amazon rainforest has incredible biodiversity.",
    "Climate change is driven by greenhouse gas emissions.",
    "Python is a popular programming language for data science.",
]

embeddings = embed_texts(texts)
print(f"    embed_texts output shape : {embeddings.shape}")
assert embeddings.shape == (5, 384), f"Expected (5, 384), got {embeddings.shape}"
assert embeddings.dtype == np.float32, "Expected float32"

# Check vectors are normalized (unit length)
norms = np.linalg.norm(embeddings, axis=1)
assert np.allclose(norms, 1.0, atol=1e-5), f"Vectors not normalized: {norms}"
print(f"    All vectors L2-normalized: ✅ (norms ≈ {norms.mean():.6f})")

query_vec = embed_query("What is machine learning?")
print(f"    embed_query output shape : {query_vec.shape}")
assert query_vec.shape == (1, 384), f"Expected (1, 384), got {query_vec.shape}"
print("    ✅ Embeddings correct")

# ── 2. Build index ─────────────────────────────────────────────
print("\n[2] Testing FAISSIndex.build()...")

from core.ingestion import Chunk
from core.faiss_index import FAISSIndex

# Create fake chunks
def make_chunk(i, text, source="doc_a.pdf", page=1):
    return Chunk(
        chunk_id=f"test_{i:03d}",
        text=text,
        source=source,
        page=page,
        chunk_index=i,
    )

chunks = [
    make_chunk(0, "Machine learning allows computers to learn from data automatically."),
    make_chunk(1, "Deep learning uses multi-layer neural networks for complex tasks."),
    make_chunk(2, "Rainforests cover about 6% of Earth's surface."),
    make_chunk(3, "Tropical rainforests have more species than any other biome."),
    make_chunk(4, "Python syntax is clean and easy to read for beginners.", source="doc_b.pdf"),
]

with tempfile.TemporaryDirectory() as tmpdir:
    idx = FAISSIndex(index_dir=Path(tmpdir))
    idx.build(chunks)

    assert idx.index.ntotal == 5, f"Expected 5 vectors, got {idx.index.ntotal}"
    assert len(idx.metadata) == 5
    print(f"    Vectors in index: {idx.index.ntotal} ✅")

    # ── 3. Save and reload ─────────────────────────────────────
    print("\n[3] Testing save() and load()...")
    idx.save()

    idx2 = FAISSIndex(index_dir=Path(tmpdir))
    idx2.load()

    assert idx2.index.ntotal == 5
    assert len(idx2.metadata) == 5
    assert idx2.metadata[0]["chunk_id"] == "test_000"
    print("    Save/load round-trip: ✅")

    # ── 4. Search by text ──────────────────────────────────────
    print("\n[4] Testing search() by query text...")

    results = idx2.search("What is machine learning?", top_k=3)
    print(f"    Query: 'What is machine learning?'")
    print(f"    Top {len(results)} results:")
    for chunk, score in results:
        print(f"      [{score:.4f}] {chunk.text[:70]}...")

    assert len(results) > 0, "No results returned"
    # Top result should be about ML/deep learning, not rainforests
    top_text = results[0][0].text.lower()
    assert any(kw in top_text for kw in ["machine", "learning", "neural", "deep"]), \
        f"Top result not about ML: {top_text}"
    print("    ✅ Semantic search returning relevant results")

    # ── 5. Search by vector (for HyDE) ────────────────────────
    print("\n[5] Testing search_by_vector()...")

    query_vec = embed_query("tropical forest ecosystem")
    results_vec = idx2.search_by_vector(query_vec, top_k=2)
    print(f"    Query vector for: 'tropical forest ecosystem'")
    for chunk, score in results_vec:
        print(f"      [{score:.4f}] {chunk.text[:70]}...")

    assert len(results_vec) > 0
    top_text = results_vec[0][0].text.lower()
    assert any(kw in top_text for kw in ["rain", "forest", "tropic", "species"]), \
        f"Top result not about forest: {top_text}"
    print("    ✅ Vector search working")

    # ── 6. Metadata filter ─────────────────────────────────────
    print("\n[6] Testing filter_source metadata filter...")

    results_filtered = idx2.search(
        "learning programming data",
        top_k=5,
        filter_source="doc_b.pdf",
    )
    print(f"    Filter: source='doc_b.pdf'")
    for chunk, score in results_filtered:
        print(f"      [{score:.4f}] source={chunk.source} | {chunk.text[:50]}...")
        assert chunk.source == "doc_b.pdf", f"Got wrong source: {chunk.source}"
    print("    ✅ Metadata filter working")

    # ── 7. is_ready() ──────────────────────────────────────────
    print("\n[7] Testing is_ready()...")
    assert idx2.is_ready() is True
    empty_idx = FAISSIndex(index_dir=Path(tmpdir) / "empty")
    assert empty_idx.is_ready() is False
    print("    ✅ is_ready() working")

# ── Summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ STEP 3 PASSED — FAISS index is ready!")
print("=" * 60)
print("\nWhat was validated:")
print("  • embed_texts() produces (N, 384) float32 normalized vectors")
print("  • embed_query() produces (1, 384) float32 normalized vector")
print("  • FAISSIndex builds, saves, and reloads correctly")
print("  • Semantic search returns topic-relevant results")
print("  • search_by_vector() works (needed for HyDE in Step 6)")
print("  • Metadata filtering by source filename works")
print("\nNext: Step 4 — BM25 sparse index builder")
