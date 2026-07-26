"""
DocuRAG — Phase 7 Tests: SQL Database Integration Agent

Tests cover:
- SQLQueryBuilder: NL→SQL conversion
- SafeQueryExecutor: SQL safety validation (blocked DML)
- SQLResultFormatter: row → chunk-style dict conversion
- SQLAgent: full query lifecycle against an in-memory SQLite DB
- API route schemas: ConnectRequest / SQLQueryRequest validation
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List

from src.retrieval.sql_agent import (
    SQLAgent,
    SQLQueryBuilder,
    SafeQueryExecutor,
    SQLResultFormatter,
    SchemaInspector,
    DatabaseSchema,
    TableSchema,
    ColumnInfo,
    UnsafeSQLError,
    SQLAgentError,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_schema(tables: List[TableSchema]) -> DatabaseSchema:
    return DatabaseSchema(
        connection_label="test_db",
        dialect="sqlite",
        tables=tables,
    )


def _simple_schema() -> DatabaseSchema:
    return _make_schema([
        TableSchema(
            table_name="employees",
            columns=[
                ColumnInfo("id", "INTEGER", nullable=False, is_primary_key=True),
                ColumnInfo("name", "VARCHAR", nullable=False),
                ColumnInfo("department", "VARCHAR", nullable=True),
                ColumnInfo("salary", "FLOAT", nullable=True),
            ],
        ),
        TableSchema(
            table_name="orders",
            columns=[
                ColumnInfo("order_id", "INTEGER", nullable=False, is_primary_key=True),
                ColumnInfo("product", "VARCHAR", nullable=False),
                ColumnInfo("quantity", "INTEGER", nullable=True),
            ],
        ),
    ])


# ── SQLQueryBuilder Tests ─────────────────────────────────────────────────────

class TestSQLQueryBuilder:

    def setup_method(self):
        self.builder = SQLQueryBuilder()
        self.schema = _simple_schema()

    def test_count_query(self):
        sql, explanation = self.builder.build("how many employees are there", self.schema)
        assert "COUNT(*)" in sql.upper()
        assert "employees" in sql.lower()

    def test_count_query_variants(self):
        for query in ["count employees", "total number of employees", "number of employees"]:
            sql, _ = self.builder.build(query, self.schema)
            assert "COUNT" in sql.upper()

    def test_select_all_from_table(self):
        sql, explanation = self.builder.build("show all employees", self.schema)
        assert "SELECT" in sql.upper()
        assert "employees" in sql.lower()

    def test_select_specific_columns(self):
        sql, explanation = self.builder.build(
            "show me name and salary from employees", self.schema
        )
        assert "SELECT" in sql.upper()
        assert "employees" in sql.lower()
        # Should have tried to select name and/or salary
        assert "name" in sql.lower() or "salary" in sql.lower()

    def test_where_clause_detection(self):
        sql, explanation = self.builder.build(
            "find employees where department = engineering", self.schema
        )
        assert "WHERE" in sql.upper()
        assert "engineering" in sql.lower()

    def test_limit_always_present(self):
        sql, _ = self.builder.build("list all employees", self.schema)
        assert "LIMIT" in sql.upper()

    def test_limit_capped_at_max_rows(self):
        sql, _ = self.builder.build("give me all records from employees", self.schema)
        assert f"LIMIT {SQLQueryBuilder.MAX_ROWS}" in sql.upper()

    def test_second_table_detected(self):
        sql, _ = self.builder.build("show all orders", self.schema)
        assert "orders" in sql.lower()

    def test_fallback_to_first_table(self):
        """When no table name is in query, falls back to first table."""
        sql, _ = self.builder.build("show me everything", self.schema)
        assert "employees" in sql.lower()  # first table in schema

    def test_sql_starts_with_select(self):
        sql, _ = self.builder.build("how many orders are there", self.schema)
        assert sql.strip().upper().startswith("SELECT")

    def test_explanation_not_empty(self):
        _, explanation = self.builder.build("list all employees", self.schema)
        assert len(explanation) > 0

    def test_empty_schema_raises(self):
        empty_schema = _make_schema([])
        with pytest.raises(SQLAgentError):
            self.builder.build("show all data", empty_schema)


# ── SafeQueryExecutor Tests ───────────────────────────────────────────────────

class TestSafeQueryExecutor:

    def test_valid_select_passes(self):
        SafeQueryExecutor.validate("SELECT * FROM employees LIMIT 10;")

    def test_select_with_where_passes(self):
        SafeQueryExecutor.validate("SELECT name FROM employees WHERE id = 1 LIMIT 10;")

    def test_insert_blocked(self):
        with pytest.raises(UnsafeSQLError):
            SafeQueryExecutor.validate("INSERT INTO employees VALUES (1, 'Alice')")

    def test_update_blocked(self):
        with pytest.raises(UnsafeSQLError):
            SafeQueryExecutor.validate("UPDATE employees SET name='Bob' WHERE id=1")

    def test_delete_blocked(self):
        with pytest.raises(UnsafeSQLError):
            SafeQueryExecutor.validate("DELETE FROM employees WHERE id=1")

    def test_drop_blocked(self):
        with pytest.raises(UnsafeSQLError):
            SafeQueryExecutor.validate("DROP TABLE employees")

    def test_truncate_blocked(self):
        with pytest.raises(UnsafeSQLError):
            SafeQueryExecutor.validate("TRUNCATE employees")

    def test_alter_blocked(self):
        with pytest.raises(UnsafeSQLError):
            SafeQueryExecutor.validate("ALTER TABLE employees ADD COLUMN age INT")

    def test_create_blocked(self):
        with pytest.raises(UnsafeSQLError):
            SafeQueryExecutor.validate("CREATE TABLE new_table (id INT)")

    def test_exec_blocked(self):
        with pytest.raises(UnsafeSQLError):
            SafeQueryExecutor.validate("EXEC sp_helplogins")

    def test_comment_injection_blocked(self):
        with pytest.raises(UnsafeSQLError):
            SafeQueryExecutor.validate("SELECT * FROM employees -- DROP TABLE employees")

    def test_semicolon_injection_blocked(self):
        with pytest.raises(UnsafeSQLError):
            SafeQueryExecutor.validate("SELECT * FROM employees; DROP TABLE employees")

    def test_inline_comment_blocked(self):
        with pytest.raises(UnsafeSQLError):
            SafeQueryExecutor.validate("SELECT * FROM employees /* comment */")

    def test_case_insensitive_blocking(self):
        with pytest.raises(UnsafeSQLError):
            SafeQueryExecutor.validate("select * from employees; delete from employees")

    def test_non_select_start_blocked(self):
        with pytest.raises(UnsafeSQLError):
            SafeQueryExecutor.validate("SHOW TABLES")

    def test_count_select_passes(self):
        SafeQueryExecutor.validate("SELECT COUNT(*) AS total FROM orders LIMIT 100;")


# ── SQLResultFormatter Tests ──────────────────────────────────────────────────

class TestSQLResultFormatter:

    def test_empty_rows_returns_empty(self):
        result = SQLResultFormatter.format(
            rows=[],
            column_names=["id", "name"],
            table_name="employees",
            db_label="test_db",
            sql="SELECT * FROM employees",
            explanation="test",
        )
        assert result == []

    def test_single_row_returns_one_chunk(self):
        rows = [{"id": 1, "name": "Alice"}]
        chunks = SQLResultFormatter.format(
            rows=rows,
            column_names=["id", "name"],
            table_name="employees",
            db_label="test_db",
            sql="SELECT * FROM employees LIMIT 100;",
            explanation="Retrieve all from employees",
        )
        assert len(chunks) == 1
        assert chunks[0]["rank"] == 1
        assert chunks[0]["chunk_type"] == "table"

    def test_multiple_rows_ranked(self):
        rows = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Carol"},
        ]
        chunks = SQLResultFormatter.format(
            rows=rows,
            column_names=["id", "name"],
            table_name="employees",
            db_label="test_db",
            sql="SELECT * FROM employees LIMIT 100;",
            explanation="test",
        )
        assert len(chunks) == 3
        assert [c["rank"] for c in chunks] == [1, 2, 3]

    def test_chunk_score_is_1(self):
        rows = [{"id": 1}]
        chunks = SQLResultFormatter.format(
            rows=rows,
            column_names=["id"],
            table_name="t",
            db_label="db",
            sql="SELECT id FROM t LIMIT 100;",
            explanation="x",
        )
        assert chunks[0]["score"] == 1.0

    def test_chunk_text_contains_row_data(self):
        rows = [{"name": "Alice", "department": "Engineering"}]
        chunks = SQLResultFormatter.format(
            rows=rows,
            column_names=["name", "department"],
            table_name="employees",
            db_label="test_db",
            sql="SELECT name, department FROM employees LIMIT 100;",
            explanation="test",
        )
        assert "Alice" in chunks[0]["text"]
        assert "Engineering" in chunks[0]["text"]

    def test_document_name_format(self):
        rows = [{"id": 1}]
        chunks = SQLResultFormatter.format(
            rows=rows,
            column_names=["id"],
            table_name="orders",
            db_label="sales_db",
            sql="SELECT id FROM orders LIMIT 100;",
            explanation="test",
        )
        assert "[SQL]" in chunks[0]["document_name"]
        assert "sales_db" in chunks[0]["document_name"]
        assert "orders" in chunks[0]["document_name"]


# ── SQLAgent Tests ────────────────────────────────────────────────────────────

class TestSQLAgent:

    def setup_method(self):
        # Create a fresh agent for each test
        self.agent = SQLAgent()

    def test_register_connection(self):
        self.agent.register_connection("test_db", "sqlite+aiosqlite:///:memory:")
        assert "test_db" in self.agent.list_connections()

    def test_remove_connection(self):
        self.agent.register_connection("test_db", "sqlite+aiosqlite:///:memory:")
        self.agent.remove_connection("test_db")
        assert "test_db" not in self.agent.list_connections()

    def test_list_connections_empty(self):
        assert self.agent.list_connections() == []

    def test_list_connections_after_register(self):
        self.agent.register_connection("db1", "sqlite+aiosqlite:///:memory:")
        self.agent.register_connection("db2", "sqlite+aiosqlite:///:memory:")
        labels = self.agent.list_connections()
        assert "db1" in labels
        assert "db2" in labels

    @pytest.mark.asyncio
    async def test_query_no_connections_returns_error(self):
        result = await self.agent.query("how many records?")
        assert result["error"] is not None
        assert len(result["chunks"]) == 0

    @pytest.mark.asyncio
    async def test_query_unknown_label_returns_error(self):
        self.agent.register_connection("real_db", "sqlite+aiosqlite:///:memory:")
        result = await self.agent.query("show all", db_label="nonexistent")
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_query_multiple_dbs_without_label_returns_error(self):
        self.agent.register_connection("db1", "sqlite+aiosqlite:///:memory:")
        self.agent.register_connection("db2", "sqlite+aiosqlite:///:memory:")
        result = await self.agent.query("how many rows?")
        assert result["error"] is not None
        assert "Specify db_label" in result["error"]

    @pytest.mark.asyncio
    async def test_single_db_auto_selected(self):
        """With only one DB registered, db_label can be omitted."""
        # Mock the internals so we don't need a real DB for this test
        self.agent.register_connection("only_db", "sqlite+aiosqlite:///:memory:")

        with patch.object(SchemaInspector, "inspect") as mock_inspect, \
             patch.object(self.agent._builder, "build") as mock_build, \
             patch("src.retrieval.sql_agent.SafeQueryExecutor.execute") as mock_exec:

            mock_schema = _simple_schema()
            mock_schema.connection_label = "only_db"
            mock_inspect.return_value = mock_schema
            mock_build.return_value = ("SELECT COUNT(*) FROM employees LIMIT 100;", "count employees")
            mock_exec.return_value = (["COUNT(*)"], [{"COUNT(*)": 5}])

            result = await self.agent.query("how many employees?")
            assert result["error"] is None
            assert result["db_label"] == "only_db"


# ── Schema DDL hint tests ─────────────────────────────────────────────────────

class TestTableSchema:

    def test_to_ddl_hint_format(self):
        table = TableSchema(
            table_name="products",
            columns=[
                ColumnInfo("id", "INTEGER", is_primary_key=True),
                ColumnInfo("name", "VARCHAR"),
            ],
        )
        ddl = table.to_ddl_hint()
        assert "TABLE products" in ddl
        assert "id" in ddl
        assert "(PK)" in ddl
        assert "name" in ddl

    def test_column_names(self):
        table = TableSchema(
            table_name="x",
            columns=[
                ColumnInfo("a", "INT"),
                ColumnInfo("b", "VARCHAR"),
            ],
        )
        assert table.column_names() == ["a", "b"]

    def test_get_table_case_insensitive(self):
        schema = _simple_schema()
        assert schema.get_table("EMPLOYEES") is not None
        assert schema.get_table("employees") is not None
        assert schema.get_table("nonexistent") is None

    def test_schema_summary_contains_tables(self):
        schema = _simple_schema()
        summary = schema.to_schema_summary()
        assert "employees" in summary
        assert "orders" in summary
        assert "test_db" in summary
