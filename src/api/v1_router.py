"""
DocuRAG API v1 - Router aggregator.

All v1 sub-routers are imported and assembled here into a single router
that is mounted at /api/v1/ by the FastAPI application factory.
"""
from fastapi import APIRouter

from src.api.routes import document_routes, health_routes, chat_routes

router = APIRouter()

# Health & diagnostics
router.include_router(health_routes.router, prefix="/health", tags=["health"])

# Document ingestion & management
router.include_router(document_routes.router, prefix="/documents", tags=["documents"])

# Chat & Retrieval
router.include_router(chat_routes.router, prefix="/chat", tags=["chat"])
