"""
DocuRAG — Phase 6: Search Router

The SearchRouter wires Phase 6 (QueryPlanner) to Phase 5 (SearchService).

Execution flow per query:
    1. QueryPlanner.plan(query)  → QueryPlan  (< 1 ms, heuristic)
    2. Route based on plan.intent:
         SQL_DATA      → SQL agent path (stub, ready for Phase 7)
         all others    → SearchService.search() with plan.strategy params
    3. Return a unified SearchResult

This is the single call-site used by the /api/v1/search endpoint.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from src.retrieval.query_planner import QueryPlan, QueryPlanner, QueryIntent
from src.retrieval.search_service import SearchService
from src.retrieval.sql_agent import sql_agent
from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)

# Singleton planner — stateless, safe to share
_planner = QueryPlanner()


@dataclass
class SearchResult:
    """Unified output from SearchRouter regardless of retrieval path taken."""
    query: str
    intent: str
    chunks: List[dict]
    latency_ms: float
    total_candidates: int
    reranked: bool
    plan: dict            # full QueryPlan serialised for transparency
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "intent": self.intent,
            "chunks": self.chunks,
            "latency_ms": self.latency_ms,
            "total_candidates": self.total_candidates,
            "reranked": self.reranked,
            "plan": self.plan,
            "error": self.error,
        }


async def _invoke_sql_agent(
    plan: QueryPlan,
    db_label: Optional[str] = None,
) -> SearchResult:
    """
    Phase 7: Route SQL-intent queries to the SQLAgent.

    The agent introspects the schema, builds a safe SELECT statement,
    executes it, and returns results in the unified CitedChunk format.
    """
    logger.info(
        "SQL intent detected — invoking Phase 7 SQL agent",
        db_label=db_label,
        query=plan.cleaned_query[:80],
    )

    result = await sql_agent.query(
        natural_language_query=plan.cleaned_query,
        db_label=db_label,
    )

    return SearchResult(
        query=plan.original_query,
        intent=plan.intent.value,
        chunks=result.get("chunks", []),
        latency_ms=result.get("latency_ms", 0.0),
        total_candidates=result.get("rows_returned", 0),
        reranked=False,
        plan={
            **plan.to_dict(),
            "sql": result.get("sql", ""),
            "sql_explanation": result.get("explanation", ""),
            "db_label": result.get("db_label", ""),
        },
        error=result.get("error"),
    )


class SearchRouter:
    """
    Phase 6: Intent-Aware Search Router.

    Combines the QueryPlanner (intent detection) with the SearchService
    (hybrid retrieval) and optional SQL agent routing.

    Usage:
        router = SearchRouter(db)
        result = await router.route(query, document_ids=[...])
        chunks = result.chunks  # list of CitedChunk dicts
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self._service = SearchService(db)

    async def route(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k_override: Optional[int] = None,
        alpha_override: Optional[float] = None,
        db_label: Optional[str] = None,
    ) -> SearchResult:
        """
        Plan and execute retrieval for the given query.

        Parameters
        ----------
        query           : Raw user query string.
        document_ids    : Optional list of document UUIDs to restrict search to.
        top_k_override  : Override the planner's top_k suggestion.
        alpha_override  : Override the planner's dense/sparse alpha.
        """
        t0 = time.perf_counter()

        # ── Phase 6: Intent detection ─────────────────────────────────────
        plan = _planner.plan(query)

        # ── Phase 7: SQL agent routing ─────────────────────────────────────
        if plan.requires_sql:
            return await _invoke_sql_agent(plan, db_label=db_label)

        # ── Phase 5: Vector + BM25 + Reranker retrieval ───────────────────
        strategy = plan.strategy
        effective_top_k = top_k_override if top_k_override is not None else strategy.top_k
        effective_alpha = alpha_override if alpha_override is not None else strategy.alpha

        try:
            result_data = await self._service.search(
                query=plan.cleaned_query,
                top_k=effective_top_k,
                alpha=effective_alpha,
                document_ids=document_ids,
                chunk_types=strategy.chunk_types,
                use_reranker=strategy.use_reranker,
            )
        except Exception as exc:
            logger.error("SearchService failed", error=str(exc), exc_info=exc)
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            return SearchResult(
                query=query,
                intent=plan.intent.value,
                chunks=[],
                latency_ms=total_ms,
                total_candidates=0,
                reranked=False,
                plan=plan.to_dict(),
                error=f"Search failed: {exc}",
            )

        total_ms = round((time.perf_counter() - t0) * 1000, 2)

        return SearchResult(
            query=query,
            intent=plan.intent.value,
            chunks=result_data.get("chunks", []),
            latency_ms=total_ms,
            total_candidates=result_data.get("total_candidates", 0),
            reranked=result_data.get("reranked", False),
            plan=plan.to_dict(),
        )
