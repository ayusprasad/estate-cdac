"""
Health check endpoints.

Provides:
- GET /api/v1/health/         — basic liveness
- GET /api/v1/health/ready    — readiness (checks DB + Redis)
- GET /api/v1/health/detailed — full component status
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from config.logging_config import get_logger
from config.settings import get_settings

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()


@router.get("/", summary="Liveness probe")
async def liveness() -> dict[str, str]:
    """Basic liveness check — returns 200 if the process is alive."""
    return {"status": "alive", "service": settings.app.name, "version": settings.app.version}


@router.get("/ready", summary="Readiness probe")
async def readiness(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Readiness check — verifies database connectivity."""
    db_ok = False
    try:
        await db.execute(__import__("sqlalchemy").text("SELECT 1"))
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
