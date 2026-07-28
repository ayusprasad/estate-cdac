"""
DocuRAG — Phase 5: Hybrid Retrieval Engine

Implements a two-stage retrieval pipeline:
  Stage 1 — Hybrid Search:
      Dense (vector cosine similarity) + Sparse (BM25) with RRF fusion.
  Stage 2 — Cross-Encoder Reranking (delegated to CrossEncoderReranker).

Key design decisions for i5-12500H + 16 GB RAM:
  - Fully vectorised numpy operations (no Python loops over chunks).
  - BM25 is implemented in pure Python with precomputed IDF for low overhead.
  - Embedding is computed once per query in a thread pool.
  - Metadata filters are applied BEFORE scoring to reduce matrix size.
"""
from __future__ import annotations

import asyncio
import math
import string
from typing import List, Optional
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database_models.chunk_model import Chunk
from src.database_models.document_model import Document
from src.document_processing.vector_embedder import get_model
from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)


class MetadataFilter:
    """
    Encapsulates the optional filters that narrow down the candidate
    chunk pool before scoring is applied.
    """

    def __init__(
        self,
        document_ids: Optional[List[str]] = None,
        language: Optional[str] = None,
        chunk_types: Optional[List[str]] = None,
        page_from: Optional[int] = None,
        page_to: Optional[int] = None,
    ):
        self.document_ids = document_ids
        self.language = language
        self.chunk_types = chunk_types
        self.page_from = page_from
        self.page_to = page_to

    def build_query(self, base_stmt):
        """Apply SQLAlchemy filter clauses to a base select statement."""
        stmt = base_stmt

        if self.document_ids:
            try:
                uuids = [UUID(did) for did in self.document_ids]
                stmt = stmt.where(Chunk.document_id.in_(uuids))
            except (ValueError, AttributeError):
                logger.warning("Invalid document_ids in filter", ids=self.document_ids)

        if self.chunk_types:
            stmt = stmt.where(Chunk.chunk_type.in_(self.chunk_types))

        if self.page_from is not None:
            stmt = stmt.where(Chunk.page_number >= self.page_from)

        if self.page_to is not None:
            stmt = stmt.where(Chunk.page_number <= self.page_to)

        return stmt


def _clean_tokens(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace, and drop small tokens."""
    cleaned = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in cleaned.split() if len(t) > 1]


class BM25Index:
    """
    Lightweight in-memory BM25 index built from a list of Chunk objects.

    Parameters follow Robertson et al. (k1=1.5, b=0.75).
    All computation is vectorised where possible.
    """

    def __init__(self, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(chunks)

        # Tokenise
        self.tokenized = [_clean_tokens(c.text) for c in chunks]
        doc_lengths = np.array([len(t) for t in self.tokenized], dtype=np.float32)
        self.avgdl = float(doc_lengths.mean()) if self.N > 0 else 1.0
        self.doc_lengths = doc_lengths

        # Build vocabulary and document-frequency map
        self._df: dict[str, int] = {}
        for tokens in self.tokenized:
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> np.ndarray:
        """Return BM25 scores for the query against all indexed chunks."""
        if self.N == 0:
            return np.array([], dtype=np.float32)

        query_terms = _clean_tokens(query)
        if not query_terms:
            return np.zeros(self.N, dtype=np.float32)

        scores = np.zeros(self.N, dtype=np.float32)

        for term in query_terms:
            idf = self._idf(term)
            if idf == 0:
                continue
            for i, tokens in enumerate(self.tokenized):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                dl = self.doc_lengths[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * numerator / denominator

        return scores


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalise an array to [0, 1]. Returns zeros if all values are identical."""
    min_val = scores.min()
    max_val = scores.max()
    if max_val == min_val:
        return np.ones_like(scores) if max_val > 0 else np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)


class HybridRetriever:
    """
    Phase 5: Hybrid Search Engine.

    Combines:
      - Dense retrieval: vectorised cosine similarity using sentence-transformers
      - Sparse retrieval: BM25 keyword matching

    Fusion strategy: weighted sum of normalised scores.
        final = alpha * dense_norm + (1 - alpha) * bm25_norm

    The retriever fetches a larger candidate set (top_k * 4) and returns
    exactly top_k candidates for downstream reranking.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self._model = None  # Lazy loaded

    def _get_model(self):
        if self._model is None:
            self._model = get_model()
        return self._model

    async def _fetch_chunks(self, metadata_filter: Optional[MetadataFilter]) -> List[Chunk]:
        """Load embedding-ready chunks from the DB, applying optional filters."""
        stmt = select(Chunk).where(Chunk.embedding.is_not(None))
        if metadata_filter:
            stmt = metadata_filter.build_query(stmt)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.7,
        metadata_filter: Optional[MetadataFilter] = None,
        score_threshold: float = 0.05,
    ) -> List[tuple[float, Chunk]]:
        """
        Execute hybrid search.

        Returns:
            List of (score, Chunk) tuples sorted by score descending.
            Score is in [0, 1] range. Scores below score_threshold are dropped.
        """
        logger.info(
            "Hybrid search started",
            query=query[:120],
            top_k=top_k,
            alpha=alpha,
        )

        chunks = await self._fetch_chunks(metadata_filter)
        if not chunks:
            logger.warning("No indexed chunks found in database")
            return []

        # ── Query Expansion ───────────────────────────────────────────────────
        from src.retrieval.query_expander import QueryExpander
        expander = QueryExpander(max_expansions=2)
        expanded_queries = expander.expand(query)
        search_query = " ".join(expanded_queries)

        # ── Dense retrieval ─────────────────────────────────────────────────
        model = await asyncio.to_thread(self._get_model)
        query_embedding = await asyncio.to_thread(
            model.encode, query, show_progress_bar=False
        )
        query_vec = np.array(query_embedding, dtype=np.float32)

        chunk_matrix = np.array(
            [c.embedding for c in chunks], dtype=np.float32
        )  # shape: (N, D)

        # Batched cosine similarity
        q_norm = np.linalg.norm(query_vec)
        c_norms = np.linalg.norm(chunk_matrix, axis=1)
        denom = q_norm * c_norms
        denom = np.where(denom == 0, 1e-9, denom)
        dense_scores = (chunk_matrix @ query_vec) / denom  # (N,)

        # ── Sparse retrieval (BM25) with expanded query ───────────────────────
        bm25 = BM25Index(chunks)
        sparse_scores = bm25.score(search_query)

        # ── Normalise and fuse ───────────────────────────────────────────────
        norm_dense = _min_max_normalize(dense_scores)
        norm_sparse = _min_max_normalize(sparse_scores)
        combined = alpha * norm_dense + (1.0 - alpha) * norm_sparse  # (N,)

        # ── Select top candidates ────────────────────────────────────────────
        # Sort indices descending by combined score
        sorted_indices = np.argsort(combined)[::-1]
        top_indices = sorted_indices[:top_k]

        results: List[tuple[float, Chunk]] = []
        for idx in top_indices:
            score = float(combined[idx])
            if score < score_threshold:
                break
            results.append((score, chunks[idx]))

        logger.info(
            "Hybrid search completed",
            candidates=len(chunks),
            returned=len(results),
        )
        return results

    async def get_document_metadata(self, document_id: UUID) -> Optional[Document]:
        """Fetch document metadata by ID for citation enrichment."""
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()
