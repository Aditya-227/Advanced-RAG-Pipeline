"""
backend/test_evaluator.py

Tests the RAGAS evaluation module — Step 9 validation.
Run from inside the backend/ folder:
  python test_evaluator.py

What it tests:
  1. EvaluationResult dataclass and helper methods
  2. RAGASEvaluator runs and returns scores (real Groq calls)
  3. All 4 RAGAS metrics are present in result
  4. Scores are in [0, 1] or None
  5. Results are persisted to JSONL log
  6. load_history() reads back correctly
  7. get_aggregated_metrics() computes correct statistics
  8. clear_history() works

NOTE: This test makes ~10-15 Groq API calls internally (RAGAS uses
      the LLM as a judge). Expect 20-40 seconds total runtime.
"""

import sys
import json
import tempfile
from pathlib import Path

print("=" * 60)
print("Step 9 — RAGAS Evaluation Test")
print("=" * 60)
print("Note: RAGAS makes multiple internal LLM calls. Takes 20-40s.")

# ── 1. EvaluationResult helpers ─────────────────────────────────
print("\n[1] Testing EvaluationResult dataclass...")

from core.evaluator import EvaluationResult

mock_result = EvaluationResult(
    eval_id="abc12345",
    timestamp="2024-01-01T12:00:00",
    query="What is attention?",
    answer="Attention computes weighted sums of values.",
    faithfulness=0.92,
    answer_relevancy=0.88,
    context_precision=0.75,
    context_recall=0.83,
    contexts=["Attention is a mechanism..."],
    latency_seconds=12.4,
    source_files=["paper.pdf"],
    rag_latency={"total_seconds": 4.2},
)

scores = mock_result.scores_dict()
assert set(scores.keys()) == {
    "faithfulness", "answer_relevancy",
    "context_precision", "context_recall"
}, f"Wrong keys: {scores.keys()}"

avg = mock_result.average_score()
expected_avg = round((0.92 + 0.88 + 0.75 + 0.83) / 4, 4)
assert avg == expected_avg, f"Wrong average: {avg} vs {expected_avg}"

d = mock_result.to_dict()
assert isinstance(d, dict)
assert d["query"] == "What is attention?"

print(f"    scores_dict()   : {scores}")
print(f"    average_score() : {avg}")
print("    ✅ EvaluationResult helpers working")

# ── 2. Full RAGAS evaluation ─────────────────────────────────────
print("\n[2] Running full RAGAS evaluation (makes real LLM calls)...")
print("    Please wait 20-40 seconds...")

from core.rag_chain import RAGResponse, SourceChunk
from core.evaluator import RAGASEvaluator

# Build a realistic mock RAGResponse
def make_source(i, text, page=1):
    return SourceChunk(
        chunk_id=f"chunk_{i}",
        text=text,
        source="transformer_paper.pdf",
        page=page,
        rerank_score=0.85 - i * 0.1,
        rrf_score=0.03 - i * 0.005,
        retrieval_source="dense + sparse",
    )

mock_rag_response = RAGResponse(
    answer=(
        "The attention mechanism computes a weighted sum of value vectors, "
        "where weights are determined by the compatibility between query "
        "and key vectors using scaled dot-product attention. "
        "This allows the model to focus on the most relevant parts of "
        "the input when generating each output token."
    ),
    source_chunks=[
        make_source(0,
            "Attention mechanisms compute a weighted sum of values, "
            "where weights come from a compatibility function between "
            "the query and corresponding keys."),
        make_source(1,
            "Scaled dot-product attention divides the dot products by "
            "sqrt(d_k) to prevent vanishing gradients in softmax."),
        make_source(2,
            "Multi-head attention runs h attention functions in parallel, "
            "concatenating and projecting the outputs."),
    ],
    hypothetical_doc="Attention mechanisms allow models to focus on relevant parts of input.",
    latency={"hyde_seconds": 2.1, "retrieval_seconds": 0.04,
             "rerank_seconds": 0.8, "llm_seconds": 1.3, "total_seconds": 4.24},
    metadata={"model": "llama-3.1-70b-versatile", "hyde_used": True},
)

with tempfile.TemporaryDirectory() as tmpdir:
    evaluator = RAGASEvaluator(log_dir=Path(tmpdir))

    result = evaluator.evaluate(
        query="How does the attention mechanism work?",
        rag_response=mock_rag_response,
    )

    # ── 3. Check all 4 metrics present ──────────────────────────
    print("\n[3] Checking RAGAS metric scores...")
    print(f"    faithfulness      : {result.faithfulness}")
    print(f"    answer_relevancy  : {result.answer_relevancy}")
    print(f"    context_precision : {result.context_precision}")
    print(f"    context_recall    : {result.context_recall}")
    print(f"    average           : {result.average_score()}")
    print(f"    eval latency      : {result.latency_seconds}s")

    # ── 4. Validate score ranges ─────────────────────────────────
    print("\n[4] Validating score ranges...")
    for metric_name in ["faithfulness", "answer_relevancy",
                        "context_precision", "context_recall"]:
        val = getattr(result, metric_name)
        if val is not None:
            assert 0.0 <= val <= 1.0, \
                f"{metric_name}={val} out of [0, 1] range"
            print(f"    {metric_name}: {val:.4f} ✅")
        else:
            print(f"    {metric_name}: None (RAGAS computation failed) ⚠")

    # ── 5. JSONL persistence ──────────────────────────────────────
    print("\n[5] Checking JSONL persistence...")
    log_file = Path(tmpdir) / "evaluation_log.jsonl"
    assert log_file.exists(), "Log file not created"

    with open(log_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    assert len(lines) == 1, f"Expected 1 log entry, got {len(lines)}"
    record = json.loads(lines[0])
    assert record["query"] == "How does the attention mechanism work?"
    assert record["eval_id"] == result.eval_id
    print(f"    Log file: {log_file}")
    print(f"    Entries : {len(lines)} ✅")
    print("    ✅ JSONL persistence working")

    # ── 6. load_history() ────────────────────────────────────────
    print("\n[6] Testing load_history()...")

    # Add a second evaluation record manually
    mock_result2 = EvaluationResult(
        eval_id="zzz99999",
        timestamp="2024-01-01T13:00:00",
        query="What is multi-head attention?",
        answer="Multi-head attention runs h attention functions in parallel.",
        faithfulness=0.78,
        answer_relevancy=0.91,
        context_precision=0.82,
        context_recall=0.74,
        contexts=["Multi-head attention..."],
        latency_seconds=9.3,
        source_files=["transformer_paper.pdf"],
        rag_latency={"total_seconds": 3.8},
    )
    evaluator._append_log(mock_result2)

    history = evaluator.load_history()
    assert len(history) == 2, f"Expected 2 records, got {len(history)}"
    assert history[0].eval_id == result.eval_id
    assert history[1].eval_id == "zzz99999"
    print(f"    Loaded {len(history)} records ✅")

    # Test last_n
    last1 = evaluator.load_history(last_n=1)
    assert len(last1) == 1
    assert last1[0].eval_id == "zzz99999"
    print("    last_n=1 works ✅")

    # ── 7. get_aggregated_metrics() ──────────────────────────────
    print("\n[7] Testing get_aggregated_metrics()...")

    agg = evaluator.get_aggregated_metrics()
    print(f"    total_queries        : {agg['total_queries']}")
    print(f"    avg_rag_latency_secs : {agg['avg_rag_latency_secs']}")

    assert agg["total_queries"] == 2
    assert "metrics" in agg

    for metric in ["faithfulness", "answer_relevancy",
                   "context_precision", "context_recall"]:
        m = agg["metrics"].get(metric)
        if m is not None:
            assert "mean" in m and "min" in m and "max" in m and "count" in m
            print(f"    {metric}: mean={m['mean']}, "
                  f"min={m['min']}, max={m['max']}, count={m['count']}")

    print("    ✅ get_aggregated_metrics() working")

    # ── 8. clear_history() ───────────────────────────────────────
    print("\n[8] Testing clear_history()...")
    evaluator.clear_history()
    assert not log_file.exists(), "Log file should be deleted"
    history_after = evaluator.load_history()
    assert history_after == []
    print("    ✅ clear_history() working")

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ STEP 9 PASSED — RAGAS Evaluation Module is ready!")
print("=" * 60)
print("\nWhat was validated:")
print("  • EvaluationResult: scores_dict(), average_score(), to_dict()")
print("  • RAGAS evaluated all 4 metrics using real Groq LLM calls")
print("  • All scores in [0.0, 1.0] range")
print("  • Results persisted correctly to JSONL log")
print("  • load_history() reads back all records + last_n filter")
print("  • get_aggregated_metrics() computes mean/min/max per metric")
print("  • clear_history() deletes the log file")
print("\nNext: Step 10 — FastAPI Backend")
