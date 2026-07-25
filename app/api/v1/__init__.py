"""
DocuRAG API v1 — Router aggregator.

All v1 sub-routers are imported and assembled here into a single router
that is mounted at /api/v1/ by the FastAPI application factory.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import documents, health

router = APIRouter()

# Health & diagnostics
router.include_router(health.router, prefix="/health", tags=["health"])

# Document ingestion & management
router.include_router(documents.router, prefix="/documents", tags=["documents"])
