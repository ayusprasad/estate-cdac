"""
DocuRAG — Phase 10: RAG Evaluation & Quality Benchmarks

Provides an end-to-end evaluation framework for the RAG pipeline.
All metrics run on CPU without any external API calls.

Metrics computed:
─────────────────
Retrieval metrics (Phase 5/6):
  - Precision@K    : fraction of retrieved chunks that are relevant
  - Recall@K       : fraction of relevant chunks that were retrieved
  - MRR            : Mean Reciprocal Rank (position of first relevant chunk)
  - NDCG@K         : Normalised Discounted Cumulative Gain

Generation metrics (Phase 8):
  - Faithfulness   : Phase 9 citation verifier score (0.0–1.0)
  - Answer length  : character count
  - Latency        : retrieval + generation ms

System metrics:
  - Total pipeline latency (ms)
  - Mode: llm | extractive
  - Chunk count

Usage:
    evaluator = RAGEvaluator()

    # Evaluate a single query response
    result = evaluator.evaluate_response(
        query="What is backpropagation?",
        answer="Backpropagation is...",
        retrieved_chunks=[...],
        relevant_chunk_ids=["uuid1", "uuid2"],   # optional ground truth
        faithfulness=0.91,
        latency_ms=350.0,
    )

    # Run batch evaluation from a test set
    report = evaluator.evaluate_batch(test_cases)

    # Get evaluation summary
    summary = evaluator.summary()
"""
from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class SingleEvalResult:
    """Evaluation result for a single query-response pair."""
    query: str
    answer_length: int
    chunks_retrieved: int
    faithfulness: float
    precision_at_k: Optional[float]     # None if no ground truth provided
    recall_at_k: Optional[float]
    mrr: Optional[float]
    ndcg_at_k: Optional[float]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    mode: str                           # "llm" | "extractive"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class BatchEvalReport:
    """Aggregated metrics over multiple evaluated queries."""
    total_queries: int
    avg_faithfulness: float
    avg_precision_at_k: Optional[float]
    avg_recall_at_k: Optional[float]
    avg_mrr: Optional[float]
    avg_ndcg_at_k: Optional[float]
    avg_total_latency_ms: float
    avg_generation_ms: float
    p95_latency_ms: float
    llm_mode_fraction: float            # fraction of queries answered by LLM
    faithfulness_distribution: Dict[str, int]   # HIGH/MEDIUM/LOW counts
    results: List[SingleEvalResult] = field(default_factory=list)


# ── Metric Functions ──────────────────────────────────────────────────────────

def _precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Precision@K: fraction of top-K retrieved that are relevant."""
    if not retrieved_ids or k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for rid in top_k if rid in relevant_set)
    return hits / k


def _recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Recall@K: fraction of relevant items that appear in top-K."""
    if not relevant_ids or k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def _mean_reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> float:
    """MRR: 1/rank of first relevant item in retrieved list."""
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    NDCG@K: Normalised Discounted Cumulative Gain.
    Binary relevance (relevant = 1, else 0).
    """
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    # DCG
    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, rid in enumerate(top_k, start=1)
        if rid in relevant_set
    )

    # Ideal DCG: all relevant items at top
    ideal_k = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_k + 1))

    if idcg == 0:
        return 0.0
    return round(dcg / idcg, 4)


def _faithfulness_verdict(score: float) -> str:
    if score >= 0.8:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


# ── RAG Evaluator ─────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Phase 10: Comprehensive RAG evaluation engine.

    Tracks results across multiple queries and produces aggregated reports.
    All computation is pure Python — no external dependencies beyond stdlib.

    Ground truth labels are optional:
    - Without ground truth: only faithfulness and latency metrics computed
    - With ground truth (relevant_chunk_ids): full precision/recall/MRR/NDCG
    """

    def __init__(self):
        self._results: List[SingleEvalResult] = []

    def evaluate_response(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        faithfulness: float,
        retrieval_latency_ms: float,
        generation_latency_ms: float,
        mode: str = "extractive",
        relevant_chunk_ids: Optional[List[str]] = None,
        k: int = 5,
    ) -> SingleEvalResult:
        """
        Evaluate a single query-response pair.

        Parameters
        ----------
        query                : User's question
        answer               : Generated answer text
        retrieved_chunks     : List of chunk dicts from search
        faithfulness         : Phase 9 grounding score (0.0–1.0)
        retrieval_latency_ms : Time for retrieval pipeline
        generation_latency_ms: Time for LLM / extractive generation
        mode                 : "llm" | "extractive"
        relevant_chunk_ids   : Ground truth relevant chunk IDs (optional)
        k                    : Cutoff for Precision@K, Recall@K, NDCG@K

        Returns
        -------
        SingleEvalResult with all computed metrics
        """
        retrieved_ids = [str(c.get("chunk_id", "")) for c in retrieved_chunks]
        total_latency = retrieval_latency_ms + generation_latency_ms

        # Retrieval metrics (only if ground truth provided)
        precision = recall = mrr = ndcg = None
        if relevant_chunk_ids:
            precision = round(_precision_at_k(retrieved_ids, relevant_chunk_ids, k), 4)
            recall = round(_recall_at_k(retrieved_ids, relevant_chunk_ids, k), 4)
            mrr = round(_mean_reciprocal_rank(retrieved_ids, relevant_chunk_ids), 4)
            ndcg = _ndcg_at_k(retrieved_ids, relevant_chunk_ids, k)

        result = SingleEvalResult(
            query=query[:200],
            answer_length=len(answer),
            chunks_retrieved=len(retrieved_chunks),
            faithfulness=round(faithfulness, 4),
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=mrr,
            ndcg_at_k=ndcg,
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            total_latency_ms=round(total_latency, 2),
            mode=mode,
        )

        self._results.append(result)

        logger.info(
            "Query evaluated",
            faithfulness=faithfulness,
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=mrr,
            ndcg_at_k=ndcg,
            total_latency_ms=total_latency,
            mode=mode,
        )

        return result

    def summary(self) -> BatchEvalReport:
        """
        Aggregate all evaluated results into a BatchEvalReport.
        Call after evaluate_response() has been called for each query.
        """
        if not self._results:
            return BatchEvalReport(
                total_queries=0,
                avg_faithfulness=0.0,
                avg_precision_at_k=None,
                avg_recall_at_k=None,
                avg_mrr=None,
                avg_ndcg_at_k=None,
                avg_total_latency_ms=0.0,
                avg_generation_ms=0.0,
                p95_latency_ms=0.0,
                llm_mode_fraction=0.0,
                faithfulness_distribution={"HIGH": 0, "MEDIUM": 0, "LOW": 0},
                results=[],
            )

        faithfulness_vals = [r.faithfulness for r in self._results]
        latency_vals = [r.total_latency_ms for r in self._results]
        gen_vals = [r.generation_latency_ms for r in self._results]

        # Optional metrics (only where ground truth was available)
        precision_vals = [r.precision_at_k for r in self._results if r.precision_at_k is not None]
        recall_vals = [r.recall_at_k for r in self._results if r.recall_at_k is not None]
        mrr_vals = [r.mrr for r in self._results if r.mrr is not None]
        ndcg_vals = [r.ndcg_at_k for r in self._results if r.ndcg_at_k is not None]

        llm_count = sum(1 for r in self._results if r.mode == "llm")

        faith_dist = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for r in self._results:
            verdict = _faithfulness_verdict(r.faithfulness)
            faith_dist[verdict] += 1

        # P95 latency
        sorted_latencies = sorted(latency_vals)
        p95_idx = max(0, int(0.95 * len(sorted_latencies)) - 1)
        p95_latency = sorted_latencies[p95_idx]

        return BatchEvalReport(
            total_queries=len(self._results),
            avg_faithfulness=round(statistics.mean(faithfulness_vals), 4),
            avg_precision_at_k=round(statistics.mean(precision_vals), 4) if precision_vals else None,
            avg_recall_at_k=round(statistics.mean(recall_vals), 4) if recall_vals else None,
            avg_mrr=round(statistics.mean(mrr_vals), 4) if mrr_vals else None,
            avg_ndcg_at_k=round(statistics.mean(ndcg_vals), 4) if ndcg_vals else None,
            avg_total_latency_ms=round(statistics.mean(latency_vals), 2),
            avg_generation_ms=round(statistics.mean(gen_vals), 2),
            p95_latency_ms=round(p95_latency, 2),
            llm_mode_fraction=round(llm_count / len(self._results), 4),
            faithfulness_distribution=faith_dist,
            results=self._results.copy(),
        )

    def clear(self) -> None:
        """Reset all accumulated evaluation results."""
        self._results.clear()

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> BatchEvalReport:
        """
        Evaluate a list of test cases and return the aggregated report.

        Each test_case dict must have:
          query                : str
          answer               : str
          retrieved_chunks     : list
          faithfulness         : float
          retrieval_latency_ms : float
          generation_latency_ms: float

        Optional:
          relevant_chunk_ids   : list (for precision/recall/MRR/NDCG)
          mode                 : "llm" | "extractive"
        """
        self.clear()
        for case in test_cases:
            self.evaluate_response(
                query=case["query"],
                answer=case.get("answer", ""),
                retrieved_chunks=case.get("retrieved_chunks", []),
                faithfulness=case.get("faithfulness", 0.0),
                retrieval_latency_ms=case.get("retrieval_latency_ms", 0.0),
                generation_latency_ms=case.get("generation_latency_ms", 0.0),
                mode=case.get("mode", "extractive"),
                relevant_chunk_ids=case.get("relevant_chunk_ids"),
            )

        report = self.summary()
        logger.info(
            "Batch evaluation complete",
            total=report.total_queries,
            avg_faithfulness=report.avg_faithfulness,
            avg_latency_ms=report.avg_total_latency_ms,
            p95_latency_ms=report.p95_latency_ms,
            faith_distribution=report.faithfulness_distribution,
        )
        return report
