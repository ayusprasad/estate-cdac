"""
DocuRAG — Phase 7: SQL Database Integration Agent

Converts natural-language queries into safe, schema-grounded SQL statements
and executes them against user-registered external databases.

Architecture
────────────
  NL Query
      │
      ▼
  SchemaInspector          ← Reads table/column names from target DB
      │
      ▼
  SQLQueryBuilder          ← Builds SQL from intent signals + schema context
      │
      ▼
  SafeQueryExecutor        ← Validates & runs SELECT-only queries
      │
      ▼
  SQLResultFormatter       ← Converts rows → citation-style chunk dicts

Design decisions for i5-12500H + 16 GB RAM
─────────────────────────────────────────────
- All database I/O is async (asyncpg / aiosqlite compatible)
- Schema introspection is cached per connection string
- Only SELECT statements are ever executed — all DML is blocked at the
  validator level before any database call is made
- Results are returned in the same CitedChunk format as vector search so
  the API layer doesn't need special-casing for SQL vs document results

Supported database backends (Phase 7)
──────────────────────────────────────
- SQLite  (development / lightweight deployments)
- PostgreSQL  (production, same server as DocuRAG's own DB is fine)

Future (Phase 8+)
──────────────────
- MySQL / MariaDB
- Microsoft SQL Server
- LLM-assisted SQL generation via llama.cpp (Phase 8)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class SQLAgentError(Exception):
    """Base exception for all SQL agent errors."""


class UnsafeSQLError(SQLAgentError):
    """Raised when the generated SQL contains disallowed statements."""


class SchemaNotFoundError(SQLAgentError):
    """Raised when no schema is available for the requested connection."""


class SQLExecutionError(SQLAgentError):
    """Raised when a query fails to execute against the target database."""


# ── Schema types ──────────────────────────────────────────────────────────────

@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False


@dataclass
class TableSchema:
    table_name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    row_count_estimate: Optional[int] = None

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def to_ddl_hint(self) -> str:
        """Return a compact DDL-like string for use as a query-builder hint."""
        cols = ", ".join(
            f"{c.name} {c.data_type}{'(PK)' if c.is_primary_key else ''}"
            for c in self.columns
        )
        return f"TABLE {self.table_name} ({cols})"


@dataclass
class DatabaseSchema:
    connection_label: str           # e.g. "sales_db", "hr_postgres"
    dialect: str                    # "sqlite" | "postgresql"
    tables: List[TableSchema] = field(default_factory=list)

    def table_names(self) -> List[str]:
        return [t.table_name for t in self.tables]

    def get_table(self, name: str) -> Optional[TableSchema]:
        for t in self.tables:
            if t.table_name.lower() == name.lower():
                return t
        return None

    def to_schema_summary(self) -> str:
        """Compact human-readable schema dump used by SQLQueryBuilder."""
        lines = [f"Database: {self.connection_label} ({self.dialect})"]
        for table in self.tables:
            lines.append(f"  {table.to_ddl_hint()}")
        return "\n".join(lines)


# ── Schema Inspector ──────────────────────────────────────────────────────────

class SchemaInspector:
    """
    Introspects a database schema at runtime using SQLAlchemy's async
    inspection API, then caches results to avoid repeated round-trips.

    Cache is in-process and lives for the lifetime of the server process.
    For production use, add a TTL or invalidation hook.
    """

    _cache: Dict[str, DatabaseSchema] = {}

    @classmethod
    async def inspect(cls, connection_url: str, label: str) -> DatabaseSchema:
        """
        Return a DatabaseSchema for the given connection.

        Parameters
        ----------
        connection_url : str
            SQLAlchemy-compatible async URL.
            Examples:
              sqlite+aiosqlite:///./data/mydb.sqlite3
              postgresql+asyncpg://user:pass@localhost/mydb
        label : str
            Short name for this database (used in citations).
        """
        if label in cls._cache:
            logger.info("Schema cache hit", label=label)
            return cls._cache[label]

        logger.info("Inspecting database schema", label=label, url=connection_url.split("@")[-1])

        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text, inspect as sa_inspect

            engine = create_async_engine(connection_url, echo=False)

            tables: List[TableSchema] = []

            async with engine.connect() as conn:
                # Use run_sync to call synchronous inspection methods
                def _sync_inspect(sync_conn):
                    inspector = sa_inspect(sync_conn)
                    result_tables = []
                    for table_name in inspector.get_table_names():
                        columns = []
                        pk_cols = set(inspector.get_pk_constraint(table_name).get("constrained_columns", []))
                        for col in inspector.get_columns(table_name):
                            columns.append(ColumnInfo(
                                name=col["name"],
                                data_type=str(col["type"]),
                                nullable=col.get("nullable", True),
                                is_primary_key=col["name"] in pk_cols,
                            ))
                        result_tables.append(TableSchema(
                            table_name=table_name,
                            columns=columns,
                        ))
                    return result_tables

                tables = await conn.run_sync(_sync_inspect)

            await engine.dispose()

            # Detect dialect from URL prefix
            dialect = "postgresql" if "postgresql" in connection_url else "sqlite"
            schema = DatabaseSchema(
                connection_label=label,
                dialect=dialect,
                tables=tables,
            )
            cls._cache[label] = schema
            logger.info(
                "Schema inspection complete",
                label=label,
                tables=schema.table_names(),
            )
            return schema

        except Exception as exc:
            logger.error("Schema inspection failed", label=label, error=str(exc))
            raise SchemaNotFoundError(f"Failed to inspect schema for '{label}': {exc}") from exc

    @classmethod
    def clear_cache(cls, label: Optional[str] = None) -> None:
        """Invalidate the schema cache (all labels or a specific one)."""
        if label:
            cls._cache.pop(label, None)
        else:
            cls._cache.clear()


# ── SQL Query Builder ─────────────────────────────────────────────────────────

# Allowed SQL keywords that can appear at the start of a statement
_ALLOWED_START_PATTERN = re.compile(r"^\s*SELECT\b", re.IGNORECASE)

# Blocked patterns — any of these cause the query to be rejected outright
_BLOCKED_PATTERNS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|MERGE"
    r"|EXEC|EXECUTE|GRANT|REVOKE|ATTACH|DETACH|PRAGMA|COPY|VACUUM"
    r"|CALL|LOAD|IMPORT|EXPORT)\b",
    re.IGNORECASE,
)

# Block comment-based SQL injection and multi-statement injection.
# A trailing ";" is valid SQL — only block ";" that is followed by more content.
_COMMENT_INJECTION = re.compile(r"(--|/\*|\*/|;\s*\S)", re.IGNORECASE)


class SQLQueryBuilder:
    """
    Heuristic-based natural language → SQL converter.

    Phase 7 uses keyword/pattern matching against the schema.
    Phase 8 will replace this with llama.cpp-powered generation.

    Current capabilities:
    - Simple SELECT queries: "show all records from orders"
    - Filtered queries:      "find employees where department = sales"
    - Count queries:         "how many products are in the inventory"
    - Column selection:      "list names and emails from customers"
    - LIMIT enforcement:     hard cap of 100 rows to protect memory
    """

    MAX_ROWS = 100

    def build(self, query: str, schema: DatabaseSchema) -> Tuple[str, str]:
        """
        Build a SQL SELECT statement from a natural language query.

        Returns
        -------
        (sql_string, explanation) tuple where explanation describes what
        the query does (shown in logs and API responses).
        """
        q_lower = query.lower()

        # ── Identify target table ─────────────────────────────────────────
        target_table: Optional[str] = None
        for table in schema.tables:
            if table.table_name.lower() in q_lower:
                target_table = table.table_name
                break

        if not target_table and schema.tables:
            # Fall back to first table with a warning
            target_table = schema.tables[0].table_name
            logger.warning(
                "No table name detected in query — defaulting to first table",
                table=target_table,
                query=query[:80],
            )

        if not target_table:
            raise SQLAgentError("No tables found in the database schema.")

        table_schema = schema.get_table(target_table)
        col_names = table_schema.column_names() if table_schema else ["*"]

        # ── Detect query type ─────────────────────────────────────────────

        # COUNT query
        if any(kw in q_lower for kw in ["how many", "count", "total number", "number of"]):
            sql = f"SELECT COUNT(*) AS total FROM {target_table} LIMIT {self.MAX_ROWS};"
            explanation = f"Count total rows in '{target_table}'"
            return sql, explanation

        # Column selection from query text
        selected_cols: List[str] = []
        for col in col_names:
            if col.lower() in q_lower:
                selected_cols.append(col)

        select_clause = ", ".join(selected_cols) if selected_cols else "*"

        # WHERE clause from "where X = Y" or "where X is Y" patterns
        where_clause = ""
        where_match = re.search(
            r"\bwhere\s+(\w+)\s+(?:=|is|equals?|like)\s+['\"]?(\w[\w\s]*?)['\"]?(?:\s+|$)",
            query,
            re.IGNORECASE,
        )
        if where_match:
            col_name_raw = where_match.group(1)
            value_raw = where_match.group(2).strip()
            # Only apply WHERE if the column exists in schema
            matched_col = next(
                (c for c in col_names if c.lower() == col_name_raw.lower()), None
            )
            if matched_col:
                where_clause = f"WHERE {matched_col} = '{value_raw}'"

        sql = (
            f"SELECT {select_clause} FROM {target_table} "
            f"{where_clause} LIMIT {self.MAX_ROWS};"
        ).strip()
        explanation = (
            f"Retrieve {select_clause} from '{target_table}'"
            + (f" {where_clause}" if where_clause else "")
            + f" (max {self.MAX_ROWS} rows)"
        )
        return sql, explanation


# ── Safe Query Executor ───────────────────────────────────────────────────────

class SafeQueryExecutor:
    """
    Validates and executes SQL queries against a target database.

    Safety rules enforced before ANY execution:
    1. Query must start with SELECT (case-insensitive)
    2. Query must not contain blocked DML/DDL keywords
    3. Query must not contain comment injection patterns (--;  /*  */)
    4. Rows returned are hard-capped at SQLQueryBuilder.MAX_ROWS
    """

    @staticmethod
    def validate(sql: str) -> None:
        """Raise UnsafeSQLError if the SQL fails safety checks."""
        if not _ALLOWED_START_PATTERN.match(sql):
            raise UnsafeSQLError(f"Only SELECT statements are permitted. Received: {sql[:100]}")

        if _BLOCKED_PATTERNS.search(sql):
            raise UnsafeSQLError(f"Query contains blocked SQL keyword: {sql[:100]}")

        if _COMMENT_INJECTION.search(sql):
            raise UnsafeSQLError(f"Query contains suspicious comment syntax: {sql[:100]}")

    @staticmethod
    async def execute(
        sql: str,
        connection_url: str,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Execute a validated SELECT query.

        Returns
        -------
        (column_names, rows) where rows is a list of dicts.
        """
        SafeQueryExecutor.validate(sql)

        logger.info("Executing SQL query", sql=sql[:200])

        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text

            engine = create_async_engine(connection_url, echo=False)
            async with engine.connect() as conn:
                result = await conn.execute(text(sql))
                column_names = list(result.keys())
                rows = [dict(zip(column_names, row)) for row in result.fetchall()]

            await engine.dispose()
            logger.info("SQL execution complete", rows_returned=len(rows), columns=column_names)
            return column_names, rows

        except UnsafeSQLError:
            raise
        except Exception as exc:
            logger.error("SQL execution failed", sql=sql[:200], error=str(exc))
            raise SQLExecutionError(f"Query execution failed: {exc}") from exc


# ── Result Formatter ──────────────────────────────────────────────────────────

class SQLResultFormatter:
    """
    Converts raw SQL result rows into the same CitedChunk dict format
    used by the vector retrieval pipeline, so the API layer is unified.
    """

    @staticmethod
    def format(
        rows: List[Dict[str, Any]],
        column_names: List[str],
        table_name: str,
        db_label: str,
        sql: str,
        explanation: str,
    ) -> List[Dict[str, Any]]:
        """
        Convert rows → list of chunk-like dicts with citation metadata.

        Each row becomes one "chunk" with:
          - rank, score, chunk_id, document_id, document_name, page_number,
            section_title, chunk_type, text
        This mirrors the CitedChunk.to_dict() output from Phase 5.
        """
        if not rows:
            return []

        chunks = []
        for rank, row in enumerate(rows, start=1):
            # Format row as readable text
            text_parts = [f"{k}: {v}" for k, v in row.items()]
            row_text = " | ".join(text_parts)

            chunks.append({
                "rank": rank,
                "score": 1.0,          # SQL results have exact match confidence
                "chunk_id": f"sql_{db_label}_{table_name}_{rank}",
                "document_id": db_label,
                "document_name": f"[SQL] {db_label}.{table_name}",
                "page_number": None,
                "section_title": f"SQL: {explanation[:80]}",
                "chunk_type": "table",
                "text": row_text,
            })

        return chunks


# ── SQL Agent ─────────────────────────────────────────────────────────────────

class SQLAgent:
    """
    Phase 7: SQL Database Integration Agent.

    Entry point for all SQL-intent queries routed by Phase 6 (SearchRouter).

    Usage
    -----
    Register a database connection once at startup or via API:
        agent = SQLAgent()
        agent.register_connection("sales_db", "sqlite+aiosqlite:///./data/sales.db")

    Then call from SearchRouter when plan.requires_sql is True:
        result = await agent.query(plan.cleaned_query, db_label="sales_db")

    The result is a list of chunk-style dicts, identical in structure to
    the output of Phase 5 vector search — no special handling needed in the
    API layer.
    """

    def __init__(self):
        # label → connection_url map
        self._connections: Dict[str, str] = {}
        self._builder = SQLQueryBuilder()

    def register_connection(self, label: str, connection_url: str) -> None:
        """
        Register an external database connection.

        Parameters
        ----------
        label          : Short identifier, e.g. "sales_db", "hr"
        connection_url : Async SQLAlchemy URL, e.g.
                         "sqlite+aiosqlite:///./data/sales.sqlite3"
                         "postgresql+asyncpg://user:pass@host/dbname"
        """
        self._connections[label] = connection_url
        # Invalidate cached schema so next query re-introspects
        SchemaInspector.clear_cache(label)
        logger.info("SQL connection registered", label=label)

    def list_connections(self) -> List[str]:
        """Return labels of all registered databases."""
        return list(self._connections.keys())

    def remove_connection(self, label: str) -> None:
        """Deregister a database connection."""
        self._connections.pop(label, None)
        SchemaInspector.clear_cache(label)
        logger.info("SQL connection removed", label=label)

    async def query(
        self,
        natural_language_query: str,
        db_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a natural language query against a registered database.

        Parameters
        ----------
        natural_language_query : str
            The user's question, e.g. "how many orders were placed in 2024?"
        db_label : str | None
            Which registered database to query. If None and only one
            database is registered, that one is used automatically.

        Returns
        -------
        Dict with keys:
          chunks          : list of chunk-style result dicts
          sql             : the generated SQL string
          explanation     : human-readable description of the query
          db_label        : which database was queried
          rows_returned   : int
          latency_ms      : float
          error           : str | None
        """
        t0 = time.perf_counter()

        # ── Resolve connection ────────────────────────────────────────────
        if db_label is None:
            if len(self._connections) == 1:
                db_label = next(iter(self._connections))
            elif len(self._connections) == 0:
                return self._error_result(
                    natural_language_query,
                    "No SQL databases have been registered. "
                    "Please connect a database via POST /api/v1/sql/connect.",
                    t0,
                )
            else:
                return self._error_result(
                    natural_language_query,
                    f"Multiple databases are registered ({', '.join(self._connections)}). "
                    "Specify db_label in the request.",
                    t0,
                )

        connection_url = self._connections.get(db_label)
        if not connection_url:
            return self._error_result(
                natural_language_query,
                f"No database registered with label '{db_label}'. "
                f"Available: {', '.join(self._connections) or 'none'}",
                t0,
            )

        try:
            # ── Inspect schema ────────────────────────────────────────────
            schema = await SchemaInspector.inspect(connection_url, db_label)

            # ── Build SQL ─────────────────────────────────────────────────
            sql, explanation = self._builder.build(natural_language_query, schema)
            logger.info("SQL generated", sql=sql, explanation=explanation)

            # ── Execute ───────────────────────────────────────────────────
            column_names, rows = await SafeQueryExecutor.execute(sql, connection_url)

            # ── Format results ────────────────────────────────────────────
            # Determine which table was targeted from the SQL
            table_match = re.search(r"\bFROM\s+(\w+)", sql, re.IGNORECASE)
            table_name = table_match.group(1) if table_match else "unknown"

            chunks = SQLResultFormatter.format(
                rows=rows,
                column_names=column_names,
                table_name=table_name,
                db_label=db_label,
                sql=sql,
                explanation=explanation,
            )

            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(
                "SQL agent query complete",
                db_label=db_label,
                rows=len(rows),
                latency_ms=latency_ms,
            )

            return {
                "chunks": chunks,
                "sql": sql,
                "explanation": explanation,
                "db_label": db_label,
                "rows_returned": len(rows),
                "latency_ms": latency_ms,
                "error": None,
            }

        except (SQLAgentError, SchemaNotFoundError) as exc:
            return self._error_result(natural_language_query, str(exc), t0)
        except Exception as exc:
            logger.error("SQL agent unexpected error", error=str(exc), exc_info=exc)
            return self._error_result(natural_language_query, f"Unexpected error: {exc}", t0)

    @staticmethod
    def _error_result(query: str, error: str, t0: float) -> Dict[str, Any]:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "chunks": [],
            "sql": "",
            "explanation": "",
            "db_label": "",
            "rows_returned": 0,
            "latency_ms": latency_ms,
            "error": error,
        }


# ── Module-level singleton ────────────────────────────────────────────────────

# One shared SQLAgent instance per process.
# Register connections via sql_agent.register_connection(label, url).
sql_agent = SQLAgent()
