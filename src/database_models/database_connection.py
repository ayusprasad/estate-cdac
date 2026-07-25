"""
DocuRAG — Async SQLAlchemy Engine & Session Factory

Provides:
- engine         : AsyncEngine bound to the configured PostgreSQL database
- AsyncSessionLocal : Session factory for dependency injection
- Base           : Declarative base for all ORM models
- get_db()       : FastAPI dependency that yields a session and commits/rolls back
- create_db_and_tables() : Called at startup to create tables that don't yet exist

Design notes:
- Uses SQLAlchemy 2.0 async API throughout
- Connection pool tuned for single-machine CPU deployment
- pgvector extension must be created before Alembic migrations run
  (handled by scripts/setup_postgres.py)
"""
from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from application_configuration.environment_settings import get_settings

settings = get_settings()

# ── Engine ───────────────────────────────────────────────────────────────────
engine = create_async_engine(
    settings.database.async_url,
    echo=settings.app.debug,  # Log SQL in debug mode
)

# ── Session factory ───────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ── Declarative base ──────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Base class for all DocuRAG ORM models."""
    pass


# ── FastAPI dependency ────────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a scoped async database session.

    - Commits on success.
    - Rolls back on any exception.
    - Always closes the session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_db_and_tables() -> None:
    """
    Create all tables defined in ORM models that don't yet exist.

    Called at application startup. In production, prefer Alembic migrations;
    this function acts as a fallback for development convenience.
    """
    # Import all models so their metadata is registered with Base
    import src.database_models.document_model  # noqa: F401
    import src.database_models.page_model  # noqa: F401
    import src.database_models.chunk_model  # noqa: F401
    import src.database_models.processing_job_model  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
