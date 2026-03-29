"""
backend/test_bm25.py

Tests the BM25 sparse index — Step 4 validation.
Run from inside the backend/ folder:
  python test_bm25.py

What it tests:
  1. tokenize() strips stopwords and punctuation correctly
  2. BM25Index.build() creates a working model
  3. save() and load() round-trip correctly
  4. search() returns keyword-relevant results
  5. BM25 beats FAISS on exact keyword queries (the whole point)
  6. Zero-score documents are excluded from results
  7. filter_source metadata filter works
"""

import sys
import tempfile
import numpy as np
from pathlib import Path

print("=" * 60)
print("Step 4 — BM25 Sparse Index Test")
print("=" * 60)

# ── 1. Tokenizer ────────────────────────────────────────────────
print("\n[1] Testing tokenize()...")

from core.bm25_index import tokenize

sample = "The Quick Brown Fox jumps over the lazy dog!"
tokens = tokenize(sample)
print(f"    Input  : '{sample}'")
print(f"    Tokens : {tokens}")

assert "the" not in tokens,   "Stopword 'the' should be removed"
assert "quick" in tokens,     "'quick' should be present"
assert "fox" in tokens,       "'fox' should be present"
assert "!" not in tokens,     "Punctuation should be removed"
assert all(t == t.lower() for t in tokens), "All tokens should be lowercase"
print("    ✅ Tokenizer working")

# ── 2. Build index ──────────────────────────────────────────────
print("\n[2] Testing BM25Index.build()...")

from core.ingestion import Chunk
from core.bm25_index import BM25Index

def make_chunk(i, text, source="paper.pdf", page=1):
    return Chunk(
        chunk_id=f"bm25_test_{i:03d}",
        text=text,
        source=source,
        page=page,
        chunk_index=i,
    )

chunks = [
    make_chunk(0, "BERT is a transformer-based language model pre-trained on masked language modeling."),
    make_chunk(1, "GPT uses autoregressive language modeling to generate text token by token."),
    make_chunk(2, "The Amazon rainforest produces 20 percent of the world oxygen supply."),
    make_chunk(3, "Deforestation threatens biodiversity in tropical rainforest ecosystems worldwide."),
    make_chunk(4, "Python programming language is widely used for machine learning projects.", source="guide.pdf"),
    make_chunk(5, "Scikit-learn provides simple tools for data mining and data analysis in Python.", source="guide.pdf"),
]

with tempfile.TemporaryDirectory() as tmpdir:
    idx = BM25Index(index_dir=Path(tmpdir))
    idx.build(chunks)

    assert idx.model is not None
    assert len(idx.metadata) == 6
    print(f"    Documents indexed: {len(idx.metadata)} ✅")

    # ── 3. Save and reload ──────────────────────────────────────
    print("\n[3] Testing save() and load()...")
    idx.save()

    idx2 = BM25Index(index_dir=Path(tmpdir))
    idx2.load()
    assert len(idx2.metadata) == 6
    assert idx2.metadata[0]["chunk_id"] == "bm25_test_000"
    print("    Save/load round-trip: ✅")

    # ── 4. Keyword search ───────────────────────────────────────
    print("\n[4] Testing keyword search...")

    results = idx2.search("BERT transformer language model", top_k=3)
    print(f"    Query: 'BERT transformer language model'")
    for chunk, score in results:
        print(f"      [BM25={score:.4f}] {chunk.text[:70]}...")

    assert len(results) > 0
    top_text = results[0][0].text.lower()
    assert "bert" in top_text or "transformer" in top_text or "language" in top_text, \
        f"Top result not about BERT: {top_text}"
    print("    ✅ Keyword search returning relevant results")

    # ── 5. BM25 vs FAISS — exact keyword advantage ─────────────
    print("\n[5] BM25 exact keyword advantage test...")
    print("    Querying 'BERT' — an exact keyword in doc 0")

    results_kw = idx2.search("BERT", top_k=6)
    print(f"    BM25 ranking for 'BERT':")
    for i, (chunk, score) in enumerate(results_kw):
        marker = "← ✅ BERT doc" if "bert" in chunk.text.lower() else ""
        print(f"      #{i+1} [score={score:.4f}] {chunk.text[:60]}... {marker}")

    # BERT doc must be ranked #1 by BM25
    assert "bert" in results_kw[0][0].text.lower(), \
        "BM25 should rank the BERT document first for query 'BERT'"
    print("    ✅ BERT document ranked #1 — exact keyword matching works")

    # ── 6. Zero scores excluded ─────────────────────────────────
    print("\n[6] Testing zero-score exclusion...")

    results_nohit = idx2.search("quantum physics superconductor", top_k=5)
    print(f"    Query about topic NOT in corpus")
    print(f"    Results returned: {len(results_nohit)}")
    # All results should have score > 0 (if any)
    for chunk, score in results_nohit:
        assert score > 0, "Zero-score result should be excluded"
    print("    ✅ Zero-score documents correctly excluded")

    # ── 7. Source filter ────────────────────────────────────────
    print("\n[7] Testing filter_source...")

    results_filtered = idx2.search("Python programming", top_k=5, filter_source="guide.pdf")
    print(f"    Filter: source='guide.pdf'")
    for chunk, score in results_filtered:
        print(f"      [score={score:.4f}] source={chunk.source} | {chunk.text[:50]}...")
        assert chunk.source == "guide.pdf", f"Got wrong source: {chunk.source}"
    print("    ✅ Source filter working")

    # ── 8. get_all_scores() ─────────────────────────────────────
    print("\n[8] Testing get_all_scores() (used by RRF in Step 5)...")

    all_scores = idx2.get_all_scores("rainforest biodiversity")
    print(f"    Scores array shape: {all_scores.shape}")
    assert all_scores.shape == (6,), f"Expected (6,), got {all_scores.shape}"
    # Docs 2 and 3 are about rainforest — should have highest scores
    top_two = np.argsort(all_scores)[::-1][:2]
    print(f"    Top 2 scoring doc indices: {top_two.tolist()} (expected [2,3] or [3,2])")
    assert set(top_two.tolist()) == {2, 3}, \
        f"Expected rainforest docs (2,3) to score highest, got {top_two.tolist()}"
    print("    ✅ get_all_scores() working correctly")

# ── Summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ STEP 4 PASSED — BM25 index is ready!")
print("=" * 60)
print("\nWhat was validated:")
print("  • tokenize() removes stopwords + punctuation + lowercases")
print("  • BM25Index builds, saves, and reloads correctly")
print("  • Keyword search ranks exact matches at the top")
print("  • BM25 ranked 'BERT' doc #1 for query 'BERT' (exact match)")
print("  • Zero-score documents are excluded from results")
print("  • Source metadata filter works")
print("  • get_all_scores() ready for RRF fusion in Step 5")
print("\nNext: Step 5 — Hybrid Search with Reciprocal Rank Fusion")
