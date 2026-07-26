"""
DocuRAG — Phase 5/6: Search API Endpoint

POST /api/v1/search

This endpoint is the primary retrieval interface.
It does NOT produce an LLM-generated answer — it returns the raw
retrieved and reranked chunks with full citations. LLM generation
will be added in Phase 7.

Design notes:
  - All filtering happens via query parameters (document_ids, language, page_from, etc.)
  - Intent detection and routing happen transparently via SearchRouter.
  - The response includes the full QueryPlan so the frontend/developer can
    see exactly why the query was routed the way it was.
"""
from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.database_models.database_connection import get_db
from src.retrieval.search_router import SearchRouter, SearchResult
from src.retrieval.query_planner import QueryPlanner
from application_configuration.logger_setup import get_logger

router = APIRouter()
logger = get_logger(__name__)

_planner = QueryPlanner()


# ── Request / Response Schemas ────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """POST /search request body."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="The user's natural language question or search query.",
        examples=["Explain backpropagation"],
    )
    document_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Restrict retrieval to these document UUIDs. "
            "Pass null/omit to search all indexed documents."
        ),
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description=(
            "Number of chunks to return. If omitted, the Query Planner "
            "chooses the optimal value based on detected intent."
        ),
    )
    alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Dense vs sparse weight. 1.0 = pure vector search, 0.0 = pure BM25. "
            "Omit to use the intent-optimised value."
        ),
    )


class ChunkCitation(BaseModel):
    """A single retrieved chunk with full citation metadata."""

    rank: int
    score: float
    chunk_id: str
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chunk_type: str
    text: str


class QueryPlanResponse(BaseModel):
    """The plan produced by the Query Planner (Phase 6)."""

    intent: str
    signals: List[str]
    requires_sql: bool
    multilingual: bool
    strategy: dict


class SearchResponse(BaseModel):
    """POST /search response."""

    query: str
    intent: str
    chunks: List[ChunkCitation]
    latency_ms: float
    total_candidates: int
    reranked: bool
    plan: QueryPlanResponse
    error: Optional[str] = None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Hybrid Search with Intent Routing",
    description=(
        "Performs a two-stage retrieval: "
        "(1) Hybrid search (dense embeddings + BM25 keyword matching), "
        "(2) Cross-encoder reranking for precision. "
        "The Query Planner (Phase 6) automatically detects intent "
        "(factual, analytical, tabular, numerical, etc.) and adjusts "
        "retrieval parameters accordingly. "
        "No LLM generation — pure retrieval with citations."
    ),
    tags=["search"],
)
async def search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """
    Hybrid retrieval endpoint (Phase 5 + Phase 6).

    Returns the top relevant chunks with full citations — no hallucinations,
    no LLM generation at this stage.
    """
    logger.info("Search request received", query=request.query[:100])

    router_instance = SearchRouter(db)

    result: SearchResult = await router_instance.route(
        query=request.query,
        document_ids=request.document_ids,
        top_k_override=request.top_k,
        alpha_override=request.alpha,
    )

    plan_dict = result.plan or {}
    plan_response = QueryPlanResponse(
        intent=plan_dict.get("intent", result.intent),
        signals=plan_dict.get("signals", []),
        requires_sql=plan_dict.get("requires_sql", False),
        multilingual=plan_dict.get("multilingual", False),
        strategy=plan_dict.get("strategy", {}),
    )

    chunks = [ChunkCitation(**c) for c in result.chunks]

    return SearchResponse(
        query=result.query,
        intent=result.intent,
        chunks=chunks,
        latency_ms=result.latency_ms,
        total_candidates=result.total_candidates,
        reranked=result.reranked,
        plan=plan_response,
        error=result.error,
    )


@router.get(
    "/plan",
    summary="Preview Query Plan (dry-run)",
    description=(
        "Returns the Query Plan that would be generated for the given query "
        "without executing any retrieval. Useful for debugging intent detection."
    ),
    tags=["search"],
)
async def preview_plan(
    query: str = Query(..., min_length=1, max_length=2048),
) -> dict:
    """Dry-run the Query Planner without executing retrieval."""
    plan = _planner.plan(query)
    return plan.to_dict()
