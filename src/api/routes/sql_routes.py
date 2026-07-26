"""
DocuRAG — Phase 7: SQL Database API Routes

Endpoints:
  POST   /api/v1/sql/connect        — Register an external SQL database
  DELETE /api/v1/sql/connect/{label}— Deregister a database
  GET    /api/v1/sql/connections    — List registered databases
  GET    /api/v1/sql/schema/{label} — Inspect and return schema
  POST   /api/v1/sql/query          — Natural language query against a DB

All query execution goes through SafeQueryExecutor which only permits
SELECT statements — DML/DDL is blocked before any DB call is made.

The /query response uses the same chunk-citation format as /search,
so the frontend doesn't need any special handling for SQL results.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.retrieval.sql_agent import sql_agent, SchemaInspector, SchemaNotFoundError
from application_configuration.logger_setup import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ── Request / Response Schemas ────────────────────────────────────────────────

class ConnectRequest(BaseModel):
    """Request body for POST /sql/connect."""
    label: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description=(
            "Short identifier for this database, e.g. 'sales_db' or 'hr'. "
            "Used in queries and citations."
        ),
        examples=["sales_db"],
    )
    connection_url: str = Field(
        ...,
        min_length=10,
        description=(
            "Async SQLAlchemy connection URL. "
            "SQLite: sqlite+aiosqlite:///./data/mydb.sqlite3 "
            "PostgreSQL: postgresql+asyncpg://user:pass@host/dbname"
        ),
        examples=["sqlite+aiosqlite:///./data/sales.sqlite3"],
    )


class ConnectResponse(BaseModel):
    label: str
    message: str
    dialect: str


class ConnectionInfo(BaseModel):
    label: str
    dialect: str


class ConnectionsResponse(BaseModel):
    connections: List[ConnectionInfo]
    total: int


class ColumnInfoResponse(BaseModel):
    name: str
    data_type: str
    nullable: bool
    is_primary_key: bool


class TableSchemaResponse(BaseModel):
    table_name: str
    columns: List[ColumnInfoResponse]


class SchemaResponse(BaseModel):
    label: str
    dialect: str
    tables: List[TableSchemaResponse]
    total_tables: int


class SQLQueryRequest(BaseModel):
    """Request body for POST /sql/query."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Natural language question about the database.",
        examples=["How many customers are registered?"],
    )
    db_label: Optional[str] = Field(
        default=None,
        description=(
            "Which registered database to query. "
            "Omit if only one database is registered."
        ),
    )


class SQLQueryResponse(BaseModel):
    query: str
    db_label: str
    sql: str
    explanation: str
    rows_returned: int
    chunks: List[dict]
    latency_ms: float
    error: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/connect",
    response_model=ConnectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register an external SQL database",
    description=(
        "Registers a database connection so DocuRAG can answer natural-language "
        "queries against it. The schema is inspected and cached on first query. "
        "Supported drivers: sqlite+aiosqlite, postgresql+asyncpg."
    ),
    tags=["sql"],
)
async def connect_database(request: ConnectRequest) -> ConnectResponse:
    """Register an external SQL database connection."""
    try:
        # Pre-validate the connection by running schema inspection now
        schema = await SchemaInspector.inspect(request.connection_url, request.label)
        sql_agent.register_connection(request.label, request.connection_url)

        logger.info(
            "SQL database registered",
            label=request.label,
            tables=schema.table_names(),
        )

        return ConnectResponse(
            label=request.label,
            message=(
                f"Database '{request.label}' registered successfully. "
                f"Found {len(schema.tables)} table(s): {', '.join(schema.table_names())}."
            ),
            dialect=schema.dialect,
        )

    except SchemaNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not connect to the database: {exc}",
        )
    except Exception as exc:
        logger.error("Database registration failed", error=str(exc), exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {exc}",
        )


@router.delete(
    "/connect/{label}",
    status_code=status.HTTP_200_OK,
    summary="Deregister a SQL database",
    tags=["sql"],
)
async def disconnect_database(label: str) -> dict:
    """Remove a registered database connection."""
    if label not in sql_agent.list_connections():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No database registered with label '{label}'.",
        )
    sql_agent.remove_connection(label)
    return {"message": f"Database '{label}' deregistered successfully."}


@router.get(
    "/connections",
    response_model=ConnectionsResponse,
    summary="List all registered SQL databases",
    tags=["sql"],
)
async def list_connections() -> ConnectionsResponse:
    """Return all registered database connection labels."""
    labels = sql_agent.list_connections()
    connections = []
    for label in labels:
        cached_schema = SchemaInspector._cache.get(label)
        dialect = cached_schema.dialect if cached_schema else "unknown"
        connections.append(ConnectionInfo(label=label, dialect=dialect))

    return ConnectionsResponse(connections=connections, total=len(connections))


@router.get(
    "/schema/{label}",
    response_model=SchemaResponse,
    summary="Inspect schema of a registered database",
    description=(
        "Returns the full table and column structure for a registered database. "
        "Results are cached in-process. Call DELETE /connect/{label} and re-register "
        "to force a refresh after schema changes."
    ),
    tags=["sql"],
)
async def get_schema(label: str) -> SchemaResponse:
    """Return the cached schema for a registered database."""
    connections = sql_agent.list_connections()
    if label not in connections:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No database registered with label '{label}'. "
                f"Available: {', '.join(connections) or 'none'}"
            ),
        )

    try:
        # SchemaInspector will use the cache if available
        connection_url = sql_agent._connections[label]
        schema = await SchemaInspector.inspect(connection_url, label)

        tables = [
            TableSchemaResponse(
                table_name=t.table_name,
                columns=[
                    ColumnInfoResponse(
                        name=c.name,
                        data_type=c.data_type,
                        nullable=c.nullable,
                        is_primary_key=c.is_primary_key,
                    )
                    for c in t.columns
                ],
            )
            for t in schema.tables
        ]

        return SchemaResponse(
            label=label,
            dialect=schema.dialect,
            tables=tables,
            total_tables=len(tables),
        )

    except SchemaNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )


@router.post(
    "/query",
    response_model=SQLQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Natural language query against a registered SQL database",
    description=(
        "Converts a natural language question into a safe SELECT SQL statement, "
        "executes it, and returns results as citation-style chunks "
        "(same format as /api/v1/search). "
        "Only SELECT queries are permitted — all DML/DDL is blocked. "
        "Results are capped at 100 rows."
    ),
    tags=["sql"],
)
async def sql_query(request: SQLQueryRequest) -> SQLQueryResponse:
    """
    Execute a natural language query against a registered SQL database.

    All terminal logs will show:
    - The generated SQL
    - Number of rows returned
    - Latency in milliseconds
    """
    logger.info(
        "SQL query request received",
        query=request.query[:100],
        db_label=request.db_label,
    )

    result = await sql_agent.query(
        natural_language_query=request.query,
        db_label=request.db_label,
    )

    if result.get("error") and not result.get("chunks"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result["error"],
        )

    return SQLQueryResponse(
        query=request.query,
        db_label=result.get("db_label", ""),
        sql=result.get("sql", ""),
        explanation=result.get("explanation", ""),
        rows_returned=result.get("rows_returned", 0),
        chunks=result.get("chunks", []),
        latency_ms=result.get("latency_ms", 0.0),
        error=result.get("error"),
    )
