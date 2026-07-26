"""
DocuRAG — Phase 5: Search Service

High-level facade that orchestrates the full retrieval pipeline:

  1. Query cleaning (lowercasing, whitespace normalisation)
  2. Hybrid search (dense + BM25, Phase 5)
  3. Cross-encoder reranking (Phase 5)
  4. Citation assembly

This is the single entry point used by API routes.
Phase 6 (QueryPlanner) wraps this service and may route
queries differently based on detected intent.
"""
from __future__ import annotations

import re
import time
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database_models.chunk_model import Chunk
from src.database_models.document_model import Document
from src.retrieval.retriever import HybridRetriever, MetadataFilter
from src.retrieval.reranker import CrossEncoderReranker
from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)

# Singleton reranker to avoid re-loading the model on every request
_reranker: Optional[CrossEncoderReranker] = None


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(top_k=5)
    return _reranker


def _clean_query(query: str) -> str:
    """
    Normalise the raw user query for retrieval.
    - Strip leading/trailing whitespace
    - Collapse internal whitespace runs
    - Remove non-printable characters
    """
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    query = "".join(ch for ch in query if ch.isprintable())
    return query


class CitedChunk:
    """Structured retrieval result with full citation metadata."""

    def __init__(
        self,
        rank: int,
        score: float,
        chunk_id: str,
        document_id: str,
        document_name: str,
        page_number: Optional[int],
        section_title: Optional[str],
        chunk_type: str,
        text: str,
    ):
        self.rank = rank
        self.score = score
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.document_name = document_name
        self.page_number = page_number
        self.section_title = section_title
        self.chunk_type = chunk_type
        self.text = text

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "chunk_type": self.chunk_type,
            "text": self.text,
        }

    def format_citation(self) -> str:
        """Human-readable citation string."""
        parts = [f"[{self.rank}] {self.document_name}"]
        if self.page_number:
            parts.append(f"Page {self.page_number}")
        if self.section_title:
            parts.append(f"§ {self.section_title}")
        return " — ".join(parts)


class SearchService:
    """
    Phase 5: Full Retrieval Pipeline Facade.

    Retrieval stages:
        query → clean → hybrid search (top 20) → cross-encoder rerank (top 5)
        → assemble CitedChunk results

    The service is stateless per-request; it receives a DB session from
    the FastAPI dependency injection system.
    """

    # Default candidate multiplier: fetch top_k * CANDIDATE_MULTIPLIER
    # before reranking to give the reranker enough signal.
    CANDIDATE_MULTIPLIER = 4

    def __init__(self, db: AsyncSession):
        self.db = db

    async def _enrich_with_document(
        self,
        chunks_with_scores: List[tuple[float, Chunk]],
    ) -> List[CitedChunk]:
        """
        Fetch document names for all unique document_ids in one DB round-trip.
        """
        if not chunks_with_scores:
            return []

        unique_doc_ids = list({chunk.document_id for _, chunk in chunks_with_scores})
        result = await self.db.execute(
            select(Document).where(Document.id.in_(unique_doc_ids))
        )
        docs = {doc.id: doc for doc in result.scalars().all()}

        cited: List[CitedChunk] = []
        for rank, (score, chunk) in enumerate(chunks_with_scores, start=1):
            doc = docs.get(chunk.document_id)
            doc_name = doc.original_filename if doc else str(chunk.document_id)
            cited.append(
                CitedChunk(
                    rank=rank,
                    score=score,
                    chunk_id=str(chunk.id),
                    document_id=str(chunk.document_id),
                    document_name=doc_name,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    chunk_type=chunk.chunk_type.value if chunk.chunk_type else "text",
                    text=chunk.text.strip(),
                )
            )
        return cited

    async def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7,
        document_ids: Optional[List[str]] = None,
        language: Optional[str] = None,
        chunk_types: Optional[List[str]] = None,
        page_from: Optional[int] = None,
        page_to: Optional[int] = None,
        use_reranker: bool = True,
    ) -> dict:
        """
        Execute the full Phase 5 retrieval pipeline.

        Returns a structured dict with:
          - query: cleaned query
          - chunks: list of CitedChunk dicts
          - latency_ms: total pipeline time
          - total_candidates: how many chunks were scored
          - reranked: whether reranking was applied
        """
        t0 = time.perf_counter()
        clean_q = _clean_query(query)

        if not clean_q:
            return {
                "query": query,
                "chunks": [],
                "latency_ms": 0,
                "total_candidates": 0,
                "reranked": False,
                "error": "Empty query after cleaning",
            }

        # ── Stage 1: Metadata filter ─────────────────────────────────────
        mfilter = MetadataFilter(
            document_ids=document_ids,
            language=language,
            chunk_types=chunk_types,
            page_from=page_from,
            page_to=page_to,
        )

        # ── Stage 2: Hybrid retrieval (fetch extra candidates for reranker) ──
        candidate_k = top_k * self.CANDIDATE_MULTIPLIER if use_reranker else top_k
        retriever = HybridRetriever(self.db)
        candidates = await retriever.search(
            query=clean_q,
            top_k=candidate_k,
            alpha=alpha,
            metadata_filter=mfilter,
        )
        total_candidates = len(candidates)

        # ── Stage 3: Cross-encoder reranking ─────────────────────────────
        reranked = False
        if use_reranker and len(candidates) > 1:
            reranker = _get_reranker()
            candidates = await reranker.rerank(clean_q, candidates, top_k=top_k)
            reranked = True
        else:
            candidates = candidates[:top_k]

        # ── Stage 4: Enrich with document metadata ────────────────────────
        cited = await self._enrich_with_document(candidates)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "Search pipeline complete",
            query=clean_q[:80],
            chunks_returned=len(cited),
            latency_ms=latency_ms,
            reranked=reranked,
        )

        return {
            "query": clean_q,
            "chunks": [c.to_dict() for c in cited],
            "latency_ms": latency_ms,
            "total_candidates": total_candidates,
            "reranked": reranked,
        }
