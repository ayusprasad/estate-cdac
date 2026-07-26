"""
DocuRAG — Retrieval Package

Phase 5: Hybrid Retrieval Engine
  - HybridRetriever: Dense (embeddings) + Sparse (BM25) search
  - CrossEncoderReranker: Cross-encoder based reranking
  - SearchService: High-level search facade

Phase 6: Query Planner & Intent Router
  - QueryPlanner: Routes queries to appropriate retrieval strategies
"""
from src.retrieval.retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.search_service import SearchService

__all__ = [
    "HybridRetriever",
    "CrossEncoderReranker",
    "SearchService",
]
