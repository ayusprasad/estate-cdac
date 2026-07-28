"""
DocuRAG — Phase 5: Cross-Encoder Reranker

Takes an initial list of (score, Chunk) candidates from hybrid search and
reranks them using a cross-encoder model for drastically improved precision.

Why cross-encoders?
  Bi-encoders (sentence-transformers) embed query and passage independently.
  Cross-encoders jointly encode (query, passage) and output a single relevance
  score — far more accurate, but ~10–50× slower per pair.

  Solution: fetch top-20 with bi-encoder, rerank with cross-encoder, return top-5.
  This gives near-reranker accuracy at acceptable latency even on CPU.

Supported backends (in order of preference):
  1. sentence-transformers CrossEncoder (default, no extra deps)
  2. Fallback: score passthrough (if model not installed or disabled)

CPU budget on i5-12500H:
  Reranking 20 pairs with ms-marco-MiniLM-L-6-v2 takes ~50–120 ms on CPU.
  This is acceptable for interactive RAG workloads.
"""
from __future__ import annotations

import asyncio
import math
from typing import List, Optional

from application_configuration.logger_setup import get_logger
from src.database_models.chunk_model import Chunk

logger = get_logger(__name__)

# Global cross-encoder cache — loaded once on first use
_cross_encoder = None
_cross_encoder_attempted = False  # Prevent repeated import failures

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _load_cross_encoder():
    """Attempt to load the cross-encoder model. Returns None if unavailable."""
    global _cross_encoder, _cross_encoder_attempted
    if _cross_encoder_attempted:
        return _cross_encoder

    try:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder reranker model (Offline Mode)", model=RERANKER_MODEL)
        try:
            _cross_encoder = CrossEncoder(RERANKER_MODEL, device="cpu", local_files_only=True)
        except Exception:
            _cross_encoder = CrossEncoder(RERANKER_MODEL, device="cpu")
        _cross_encoder_attempted = True
        logger.info("Cross-encoder model loaded successfully")
    except Exception as exc:
        logger.warning(
            "Cross-encoder model unavailable — reranker disabled, using score passthrough",
            error=str(exc),
        )
        _cross_encoder = None

    return _cross_encoder


class CrossEncoderReranker:
    """
    Phase 5: Cross-Encoder Reranker.

    Usage pattern:
        # Hybrid retriever returns top_k * 4 = 20 candidates
        candidates = await retriever.search(query, top_k=20)
        # Reranker narrows to top 5 with higher precision
        top5 = await reranker.rerank(query, candidates, top_k=5)

    Graceful degradation:
        If the cross-encoder is not installed or fails to load, the reranker
        falls back to returning candidates sorted by their original hybrid score.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        # Model is lazy-loaded on first rerank() call to avoid event-loop conflicts.
        # Call _load_cross_encoder() synchronously here only if safe to do so.
        try:
            _load_cross_encoder()
        except Exception:
            pass  # Will be retried on first rerank() call

    async def rerank(
        self,
        query: str,
        candidates: List[tuple[float, Chunk]],
        top_k: Optional[int] = None,
    ) -> List[tuple[float, Chunk]]:
        """
        Rerank candidate (score, chunk) pairs using the cross-encoder.

        Args:
            query:      The user's original query string.
            candidates: Output of HybridRetriever.search() — (score, Chunk) tuples.
            top_k:      How many top results to return (defaults to self.top_k).

        Returns:
            Sorted list of (rerank_score, Chunk) tuples, descending.
        """
        k = top_k if top_k is not None else self.top_k
        if not candidates:
            return []

        model = await asyncio.to_thread(_load_cross_encoder)

        if model is None:
            # Fallback: return candidates as-is, truncated to top_k
            logger.debug("Reranker unavailable — using hybrid scores directly")
            return candidates[:k]

        # Build (query, passage) pairs for cross-encoder
        pairs = [(query, chunk.text) for _, chunk in candidates]

        # Run cross-encoder inference in thread pool (CPU-bound)
        try:
            raw_scores = await asyncio.to_thread(model.predict, pairs)
        except Exception as exc:
            logger.warning("Cross-encoder inference failed — falling back", error=str(exc))
            return candidates[:k]

        # Convert raw logits to sigmoid scores
        sigmoid_scores = [1.0 / (1.0 + math.exp(-s)) for s in raw_scores.tolist()]

        # Pair scores with chunks and sort
        reranked = sorted(
            zip(sigmoid_scores, [c for _, c in candidates]),
            key=lambda x: x[0],
            reverse=True,
        )
        logger.info(
            "Reranking complete",
            input_candidates=len(candidates),
            returned=min(k, len(reranked)),
        )
        return reranked[:k]
