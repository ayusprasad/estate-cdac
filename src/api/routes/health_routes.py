"""
Health check endpoints.

Provides:
- GET /api/v1/health/         — basic liveness
- GET /api/v1/health/ready    — readiness (checks DB)
- GET /api/v1/health/detailed — full component status
"""
from __future__ import annotations

from typing import Any

import sqlalchemy
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.database_models.database_connection import get_db
from application_configuration.logger_setup import get_logger
from application_configuration.environment_settings import get_settings

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()


@router.get("/", summary="Liveness probe")
async def liveness() -> dict[str, str]:
    """Basic liveness check — returns 200 if the process is alive."""
    return {
        "status": "alive",
        "service": settings.app.name,
        "version": settings.app.version,
    }


@router.get("/ready", summary="Readiness probe")
async def readiness(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Readiness check — verifies database connectivity."""
    db_ok = False
    try:
        await db.execute(sqlalchemy.text("SELECT 1"))
        db_ok = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Database health check failed", exc_info=exc)

    status = "ready" if db_ok else "degraded"
    return {
        "status": status,
        "components": {
            "database": "ok" if db_ok else "error",
        },
    }


@router.get("/detailed", summary="Detailed component status")
async def detailed_health(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Return detailed health status of all system components."""
    components: dict[str, Any] = {}

    # Database
    try:
        result = await db.execute(sqlalchemy.text("SELECT version()"))
        pg_version = result.scalar()
        components["database"] = {"status": "ok", "version": pg_version}
    except Exception as exc:  # noqa: BLE001
        components["database"] = {"status": "error", "detail": str(exc)}

    # pgvector extension
    try:
        result = await db.execute(
            sqlalchemy.text(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
        )
        vec_version = result.scalar()
        if vec_version:
            components["pgvector"] = {"status": "ok", "version": vec_version}
        else:
            components["pgvector"] = {"status": "not_installed"}
    except Exception as exc:  # noqa: BLE001
        components["pgvector"] = {"status": "error", "detail": str(exc)}

    # Storage directories
    storage_ok = all([
        settings.storage.raw_docs_dir.exists(),
        settings.storage.processed_docs_dir.exists(),
        settings.storage.temp_dir.exists(),
    ])
    components["storage"] = {
        "status": "ok" if storage_ok else "error",
        "raw_dir": str(settings.storage.raw_docs_dir),
        "processed_dir": str(settings.storage.processed_docs_dir),
    }

    # LLM model file presence
    llm_ready = settings.llm.model_path.exists()
    components["llm"] = {
        "status": "ready" if llm_ready else "model_not_found",
        "model_path": str(settings.llm.model_path),
        "note": "Place a GGUF model file at the configured path" if not llm_ready else None,
    }

    overall_status = (
        "ready"
        if components["database"]["status"] == "ok"
        else "degraded"
    )

    return {
        "status": overall_status,
        "service": settings.app.name,
        "version": settings.app.version,
        "environment": settings.app.env,
        "components": components,
    }
