"""
DocuRAG API v1 - Router aggregator.

All v1 sub-routers are imported and assembled here into a single router
that is mounted at /api/v1/ by the FastAPI application factory.

Phase  Module         Prefix              Tags
─────  ──────         ──────              ────
0-1    health         /health             health
0-1    documents      /documents          documents
4      chat           /chat               chat        ← Phase 8/9 integrated
5-6    search         /search             search
7      sql            /sql                sql
10     eval           /eval               eval        ← Phase 10
"""
from fastapi import APIRouter

from src.api.routes import (
    document_routes,
    health_routes,
    chat_routes,
    search_routes,
    sql_routes,
    eval_routes,
)

router = APIRouter()

# Health & diagnostics
router.include_router(health_routes.router, prefix="/health", tags=["health"])

# Document ingestion & management
router.include_router(document_routes.router, prefix="/documents", tags=["documents"])

# Phase 8+9: Full RAG (retrieve → generate → verify)
router.include_router(chat_routes.router, prefix="/chat", tags=["chat"])

# Phase 5+6: Hybrid Search + Intent Router
router.include_router(search_routes.router, prefix="/search", tags=["search"])

# Phase 7: SQL Database Integration Agent
router.include_router(sql_routes.router, prefix="/sql", tags=["sql"])

# Phase 10: Evaluation & Quality Benchmarks
router.include_router(eval_routes.router, prefix="/eval", tags=["eval"])

