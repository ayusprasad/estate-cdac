"""
DocuRAG — Alembic Environment Script

Configures Alembic for async SQLAlchemy with PostgreSQL.
Reads the database URL from application settings (respects .env).
Imports all ORM models so their metadata is available for autogenerate.
"""
from __future__ import annotations

import asyncio
from logging.application_configuration import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# Load DocuRAG settings and models
from application_configuration.environment_settings import get_settings
from src.database_models.database_connection import Base

# Import all models to register their metadata with Base
import src.database_models.document_model  # noqa: F401
import src.database_models.page_model  # noqa: F401
import src.database_models.chunk_model  # noqa: F401
import src.database_models.processing_job_model  # noqa: F401

settings = get_settings()

# Alembic Config object — provides access to alembic.ini values
application_configuration = context.application_configuration

# Override sqlalchemy.url from application settings
application_configuration.set_main_option("sqlalchemy.url", settings.database.async_url)

# Setup Python logging from alembic.ini
if application_configuration.config_file_name is not None:
    fileConfig(application_configuration.config_file_name)

# Target metadata for autogenerate support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (no live DB connection needed)."""
    url = application_configuration.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: object) -> None:
    """Execute migrations within a connection context."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_schemas=True,
        # Only include the 'docurag' schema in autogenerate
        include_name=lambda name, type_, parent_names: (
            name in ("docurag", "audit") if type_ == "schema" else True
        ),
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode with an async engine."""
    connectable = async_engine_from_config(
        application_configuration.get_section(application_configuration.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
