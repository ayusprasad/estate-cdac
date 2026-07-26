"""
DocuRAG — Phase 10: Evaluation API Routes

POST /api/v1/eval/query    — Evaluate a single query-response pair
POST /api/v1/eval/batch    — Evaluate a batch of test cases
GET  /api/v1/eval/report   — Get accumulated evaluation report
DELETE /api/v1/eval/report — Reset the accumulator

Terminal logs show all metrics as they are computed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from src.llm.evaluator import RAGEvaluator, SingleEvalResult, BatchEvalReport
from application_configuration.logger_setup import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Module-level evaluator accumulates results across requests for the session
_evaluator = RAGEvaluator()


# ── Request / Response Models ─────────────────────────────────────────────────

class EvalQueryRequest(BaseModel):
    query: str = Field(..., description="User question")
    answer: str = Field(..., description="Generated answer to evaluate")
    retrieved_chunks: List[Dict[str, Any]] = Field(...,
        description="Chunk list from search pipeline")
    faithfulness: float = Field(..., ge=0.0, le=1.0,
        description="Faithfulness score from Phase 9 CitationVerifier")
    retrieval_latency_ms: float = Field(default=0.0)
    generation_latency_ms: float = Field(default=0.0)
    mode: str = Field(default="extractive", description="'llm' or 'extractive'")
    relevant_chunk_ids: Optional[List[str]] = Field(
        default=None,
        description="Ground truth chunk IDs for precision/recall computation"
    )


class EvalQueryResponse(BaseModel):
    query: str
    faithfulness: float
    faithfulness_verdict: str
    precision_at_k: Optional[float]
    recall_at_k: Optional[float]
    mrr: Optional[float]
    ndcg_at_k: Optional[float]
    total_latency_ms: float
    mode: str
    answer_length: int
    chunks_retrieved: int


class BatchTestCase(BaseModel):
    query: str
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    faithfulness: float
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    mode: str = "extractive"
    relevant_chunk_ids: Optional[List[str]] = None


class EvalBatchRequest(BaseModel):
    test_cases: List[BatchTestCase] = Field(..., min_length=1)


class FaithfulnessDistribution(BaseModel):
    HIGH: int
    MEDIUM: int
    LOW: int


class EvalReportResponse(BaseModel):
    total_queries: int
    avg_faithfulness: float
    avg_precision_at_k: Optional[float]
    avg_recall_at_k: Optional[float]
    avg_mrr: Optional[float]
    avg_ndcg_at_k: Optional[float]
    avg_total_latency_ms: float
    avg_generation_ms: float
    p95_latency_ms: float
    llm_mode_fraction: float
    faithfulness_distribution: FaithfulnessDistribution


def _verdict(score: float) -> str:
    if score >= 0.8:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    return "LOW"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=EvalQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate a single query-response pair",
    tags=["eval"],
)
async def evaluate_query(request: EvalQueryRequest) -> EvalQueryResponse:
    """
    Evaluate retrieval quality and generation faithfulness for one Q&A pair.
    Results accumulate in the session evaluator (visible via GET /eval/report).
    """
    result: SingleEvalResult = _evaluator.evaluate_response(
        query=request.query,
        answer=request.answer,
        retrieved_chunks=request.retrieved_chunks,
        faithfulness=request.faithfulness,
        retrieval_latency_ms=request.retrieval_latency_ms,
        generation_latency_ms=request.generation_latency_ms,
        mode=request.mode,
        relevant_chunk_ids=request.relevant_chunk_ids,
    )

    return EvalQueryResponse(
        query=result.query,
        faithfulness=result.faithfulness,
        faithfulness_verdict=_verdict(result.faithfulness),
        precision_at_k=result.precision_at_k,
        recall_at_k=result.recall_at_k,
        mrr=result.mrr,
        ndcg_at_k=result.ndcg_at_k,
        total_latency_ms=result.total_latency_ms,
        mode=result.mode,
        answer_length=result.answer_length,
        chunks_retrieved=result.chunks_retrieved,
    )


@router.post(
    "/batch",
    response_model=EvalReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate a batch of test cases",
    description=(
        "Run the evaluator over a list of test cases. Replaces any previously "
        "accumulated results in the session evaluator. "
        "Provide relevant_chunk_ids for precision/recall/MRR/NDCG computation."
    ),
    tags=["eval"],
)
async def evaluate_batch(request: EvalBatchRequest) -> EvalReportResponse:
    """Batch evaluation endpoint."""
    test_cases = [tc.model_dump() for tc in request.test_cases]
    report: BatchEvalReport = _evaluator.evaluate_batch(test_cases)

    return EvalReportResponse(
        total_queries=report.total_queries,
        avg_faithfulness=report.avg_faithfulness,
        avg_precision_at_k=report.avg_precision_at_k,
        avg_recall_at_k=report.avg_recall_at_k,
        avg_mrr=report.avg_mrr,
        avg_ndcg_at_k=report.avg_ndcg_at_k,
        avg_total_latency_ms=report.avg_total_latency_ms,
        avg_generation_ms=report.avg_generation_ms,
        p95_latency_ms=report.p95_latency_ms,
        llm_mode_fraction=report.llm_mode_fraction,
        faithfulness_distribution=FaithfulnessDistribution(
            **report.faithfulness_distribution
        ),
    )


@router.get(
    "/report",
    response_model=EvalReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Get accumulated evaluation report",
    tags=["eval"],
)
async def get_report() -> EvalReportResponse:
    """Return aggregated metrics across all queries evaluated in this session."""
    report: BatchEvalReport = _evaluator.summary()
    return EvalReportResponse(
        total_queries=report.total_queries,
        avg_faithfulness=report.avg_faithfulness,
        avg_precision_at_k=report.avg_precision_at_k,
        avg_recall_at_k=report.avg_recall_at_k,
        avg_mrr=report.avg_mrr,
        avg_ndcg_at_k=report.avg_ndcg_at_k,
        avg_total_latency_ms=report.avg_total_latency_ms,
        avg_generation_ms=report.avg_generation_ms,
        p95_latency_ms=report.p95_latency_ms,
        llm_mode_fraction=report.llm_mode_fraction,
        faithfulness_distribution=FaithfulnessDistribution(
            **report.faithfulness_distribution
        ),
    )


@router.delete(
    "/report",
    status_code=status.HTTP_200_OK,
    summary="Reset evaluation accumulator",
    tags=["eval"],
)
async def reset_report() -> dict:
    """Clear all accumulated evaluation results for a fresh benchmark run."""
    _evaluator.clear()
    return {"message": "Evaluation results cleared."}
