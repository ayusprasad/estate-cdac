"""
DocuRAG — FastAPI Application Entry Point

Initialises the FastAPI application, registers all routers,
configures middleware (CORS, request ID), and sets up lifespan
events for database connections and logging.

Design decisions:
- Lifespan context manager (not deprecated on_startup/on_shutdown)
- Request ID middleware for end-to-end traceability
- Health check endpoint returns DB + service status
- All routers are versioned under /api/v1/
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import router as api_v1_router
from app.models.database import create_db_and_tables, engine
from config.logging_config import configure_logging, get_logger
from config.settings import get_settings

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup (logging, DB init) and shutdown (DB disposal) cleanly.
    """
    # ── Startup ────────────────────────────────────────────
    configure_logging(
        log_level=settings.logging.level,
        log_format=settings.logging.format,
    )
    logger.info(
        "Starting DocuRAG",
        version=settings.app.version,
        env=settings.app.env,
        debug=settings.app.debug,
    )
    await create_db_and_tables()
    logger.info("Database initialised")

    yield  # Application is running

    # ── Shutdown ───────────────────────────────────────────
    logger.info("Shutting down DocuRAG")
    await engine.dispose()


def create_application() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""
    app = FastAPI(
        title="DocuRAG API",
        description=(
            "Enterprise AI Document Intelligence & RAG System. "
            "Process PDFs, Word, Excel, images, and SQL databases. "
            "Answers are always grounded in cited source documents."
        ),
        version=settings.app.version,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS middleware ──────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=settings.api.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID middleware ───────────────────────────────
    @app.middleware("http")
    async def add_request_id(request: Request, call_next: Any) -> Response:
        """Inject a unique X-Request-ID header into every request/response."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        import structlog
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start_time = time.monotonic()
        response = await call_next(request)
        duration_ms = round((time.monotonic() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(duration_ms)
        logger.debug(
            "Request completed",
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response

    # ── Global exception handler ─────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled exception",
            path=str(request.url.path),
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred. Please check the logs."},
        )

    # ── Routers ───────────────────────────────────────────
    app.include_router(api_v1_router, prefix="/api/v1")

    # ── Root endpoint ──────────────────────────────────────
    @app.get("/", tags=["root"])
    async def root() -> dict[str, str]:
        """Service liveness check."""
        return {
            "service": "DocuRAG",
            "version": settings.app.version,
            "status": "operational",
            "docs": "/api/docs",
        }

    return app


# Module-level app instance consumed by uvicorn
app = create_application()

# Fix forward reference in middleware
from typing import Any  # noqa: E402 (after app creation to avoid circular import)
