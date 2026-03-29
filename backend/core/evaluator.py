"""
backend/core/evaluator.py

WHAT THIS DOES:
  Evaluates every RAG query-answer pair using RAGAS metrics.
  Persists scores to disk so the dashboard can show history.

HOW RAGAS WORKS (interview explanation):
  RAGAS (Retrieval Augmented Generation Assessment) evaluates RAG
  pipelines without needing human-labeled ground truth answers.
  It uses an LLM-as-judge approach:

  Faithfulness:
    - Breaks the answer into atomic claims
    - Checks each claim against the retrieved context
    - Score = claims_supported_by_context / total_claims
    - Catches hallucinations: if the LLM says something not in context → low score

  Answer Relevancy:
    - Generates N questions that the answer could be answering
    - Embeds those questions and the original question
    - Score = avg cosine similarity (generated questions, original question)
    - Catches off-topic answers: if answer doesn't address the query → low score

  Context Precision:
    - Checks which retrieved chunks were actually useful for the answer
    - Score = useful_chunks_ranked_high / total_retrieved_chunks
    - Measures retrieval precision: are top chunks relevant?

  Context Recall:
    - Checks if the ground truth answer can be attributed to retrieved chunks
    - Requires a reference answer — we use the LLM's answer as reference
    - Score = attributable_sentences / total_sentences_in_reference

  WHY NO HUMAN LABELS?
    Traditional evaluation needs humans to write ground truth Q&A pairs.
    RAGAS uses the LLM itself to judge quality — scalable to any dataset.

STORAGE:
  data/metrics/evaluation_log.jsonl  — one JSON record per query
  Each record: {timestamp, query, answer, scores, latency, sources}
  JSONL = JSON Lines = one JSON object per line, easy to stream/append
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import get_settings
from core.rag_chain import RAGResponse


# ── Score data model ──────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """
    RAGAS scores for one query-answer pair.

    All scores are floats in [0.0, 1.0].
    None means the metric could not be computed (e.g., API error).

    eval_id     : unique ID for this evaluation record
    timestamp   : ISO 8601 string
    query       : the user's original question
    answer      : the RAG-generated answer
    faithfulness: fraction of answer claims supported by context
    answer_relevancy: how well the answer addresses the query
    context_precision: fraction of retrieved chunks that are relevant
    context_recall: fraction of reference answer attributable to chunks
    contexts    : the retrieved chunk texts used for evaluation
    latency_seconds: how long RAGAS evaluation took
    """
    eval_id: str
    timestamp: str
    query: str
    answer: str
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]
    contexts: List[str]
    latency_seconds: float
    source_files: List[str]
    rag_latency: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)

    def scores_dict(self) -> dict:
        """Returns just the metric scores — used by the dashboard."""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }

    def average_score(self) -> Optional[float]:
        """Mean of all non-None scores — overall quality indicator."""
        scores = [s for s in self.scores_dict().values() if s is not None]
        return round(sum(scores) / len(scores), 4) if scores else None


# ── RAGAS evaluator ───────────────────────────────────────────────────────────

class RAGASEvaluator:
    """
    Evaluates RAG responses using RAGAS metrics and persists results.

    RAGAS needs:
      - question    : the user query
      - answer      : the RAG-generated answer
      - contexts    : list of retrieved chunk texts (strings)
      - ground_truth: a reference answer (we use the RAG answer itself
                      as a proxy — not ideal but works without labels)

    Usage:
        evaluator = RAGASEvaluator()
        result = evaluator.evaluate(query, rag_response)
        print(result.faithfulness)   # e.g., 0.87
    """

    def __init__(self, log_dir: Optional[Path] = None):
        cfg = get_settings()
        self.log_dir = Path(log_dir or cfg.metrics_path)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "evaluation_log.jsonl"

        # RAGAS uses an LLM internally for faithfulness + answer_relevancy
        # We reuse Groq here — same API key, no extra cost
        self._llm = ChatGroq(
            api_key=cfg.groq_api_key,
            model_name=cfg.groq_model_name,
            temperature=0.0,     # deterministic for evaluation
            max_tokens=1024,
        )

        # RAGAS uses embeddings internally for answer_relevancy
        # We reuse the same HuggingFace model — no extra download
        self._embeddings = None

    def evaluate(
        self,
        query: str,
        rag_response: RAGResponse,
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Run RAGAS evaluation on a query-answer pair.

        Args:
            query        : the user's original question
            rag_response : the RAGResponse from RAGChain.query()
            ground_truth : optional reference answer
                           if None, we use the RAG answer as its own
                           reference (proxy evaluation — less accurate
                           for context_recall but workable)

        Returns:
            EvaluationResult with all 4 RAGAS scores

        NOTE ON SPEED:
          RAGAS makes multiple LLM calls internally per metric.
          Expect 5-15 seconds per evaluation call on Groq free tier.
          This is why we evaluate asynchronously in the API (Step 10).
        """
        t0 = time.time()
        eval_id = str(uuid.uuid4())[:8]

        print(f"\n[RAGAS] Evaluating query: '{query[:60]}...'")

        # ── Build RAGAS dataset ───────────────────────────────────────────────
        contexts = [src.text for src in rag_response.source_chunks]
        answer   = rag_response.answer

        # ground_truth: RAGAS context_recall needs a reference answer.
        # Without human labels we use the generated answer itself.
        # This makes context_recall measure "can the context reproduce
        # our answer" rather than "can the context reproduce the truth".
        gt = ground_truth or answer

        ragas_dataset = Dataset.from_dict({
            "question"    : [query],
            "answer"      : [answer],
            "contexts"    : [contexts],
            "ground_truth": [gt],
        })

        # ── Run RAGAS evaluation ──────────────────────────────────────────────
        scores = {
            "faithfulness"      : None,
            "answer_relevancy"  : None,
            "context_precision" : None,
            "context_recall"    : None,
        }

        # Lazy-load embeddings on first evaluation call
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings as HFE
            self._embeddings = HFE(model_name=cfg.embedding_model_name)
        try:
            result = evaluate(
                dataset=ragas_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
                llm=self._llm,
                embeddings=self._embeddings,
                raise_exceptions=False,   # don't crash on partial failures
            )

            # result is a dict-like object — extract scores safely
            result_dict = result.to_pandas().iloc[0].to_dict()

            scores["faithfulness"]       = self._safe_float(result_dict.get("faithfulness"))
            scores["answer_relevancy"]   = self._safe_float(result_dict.get("answer_relevancy"))
            scores["context_precision"]  = self._safe_float(result_dict.get("context_precision"))
            scores["context_recall"]     = self._safe_float(result_dict.get("context_recall"))

        except Exception as e:
            print(f"[RAGAS] ⚠ Evaluation error: {e}")
            print("[RAGAS] Returning None scores — pipeline continues.")
            # Scores stay None — dashboard will show "N/A"

        latency = round(time.time() - t0, 2)

        print(f"[RAGAS] Scores: "
              f"faithfulness={scores['faithfulness']} | "
              f"relevancy={scores['answer_relevancy']} | "
              f"precision={scores['context_precision']} | "
              f"recall={scores['context_recall']} | "
              f"({latency}s)")

        # ── Build result object ───────────────────────────────────────────────
        eval_result = EvaluationResult(
            eval_id=eval_id,
            timestamp=datetime.utcnow().isoformat(),
            query=query,
            answer=answer,
            faithfulness=scores["faithfulness"],
            answer_relevancy=scores["answer_relevancy"],
            context_precision=scores["context_precision"],
            context_recall=scores["context_recall"],
            contexts=contexts,
            latency_seconds=latency,
            source_files=list({src.source for src in rag_response.source_chunks}),
            rag_latency=rag_response.latency,
        )

        # ── Persist to JSONL ──────────────────────────────────────────────────
        self._append_log(eval_result)

        return eval_result

    def _safe_float(self, value) -> Optional[float]:
        """Convert a RAGAS score to float, returning None on failure."""
        if value is None:
            return None
        try:
            f = float(value)
            # RAGAS can sometimes return NaN
            import math
            return None if math.isnan(f) else round(f, 4)
        except (TypeError, ValueError):
            return None

    def _append_log(self, result: EvaluationResult) -> None:
        """Append one evaluation record to the JSONL log file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    # ── History and aggregation ────────────────────────────────────────────────

    def load_history(self, last_n: Optional[int] = None) -> List[EvaluationResult]:
        """
        Load evaluation history from the JSONL log.

        Args:
            last_n : if set, return only the most recent N records

        Returns:
            List of EvaluationResult objects, oldest first
        """
        if not self.log_file.exists():
            return []

        results = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    results.append(EvaluationResult(**data))
                except Exception:
                    continue  # skip malformed lines

        if last_n is not None:
            results = results[-last_n:]

        return results

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics across all evaluations.

        Returns a dict with mean, min, max for each metric,
        plus total query count and average RAG latency.
        Used by the GET /metrics API endpoint.
        """
        history = self.load_history()

        if not history:
            return {
                "total_queries": 0,
                "metrics": {},
                "message": "No evaluations yet. Run some queries first.",
            }

        metric_names = [
            "faithfulness", "answer_relevancy",
            "context_precision", "context_recall",
        ]

        aggregated = {}
        for metric in metric_names:
            values = [
                getattr(r, metric) for r in history
                if getattr(r, metric) is not None
            ]
            if values:
                aggregated[metric] = {
                    "mean" : round(sum(values) / len(values), 4),
                    "min"  : round(min(values), 4),
                    "max"  : round(max(values), 4),
                    "count": len(values),
                }
            else:
                aggregated[metric] = None

        # Average total RAG latency
        latencies = [
            r.rag_latency.get("total_seconds", 0)
            for r in history
            if r.rag_latency
        ]
        avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0

        return {
            "total_queries"        : len(history),
            "avg_rag_latency_secs" : avg_latency,
            "metrics"              : aggregated,
        }

    def clear_history(self) -> None:
        """Delete the evaluation log (useful for testing)."""
        if self.log_file.exists():
            self.log_file.unlink()


# ── Module-level singleton ─────────────────────────────────────────────────────

_evaluator: Optional[RAGASEvaluator] = None


def get_evaluator() -> RAGASEvaluator:
    """Returns the global RAGASEvaluator singleton."""
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGASEvaluator()
    return _evaluator
