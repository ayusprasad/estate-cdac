"""
DocuRAG — Initial Database Migration

Creates the core schema, enums, and tables for Phase 0/1.
Includes: documents, pages, chunks, processing_jobs

Revision ID: 001_initial
Revises: 
Create Date: 2026-07-25
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector  # type: ignore[import]
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all core tables."""

    # ── Create schemas ────────────────────────────────────────────────────────
    op.execute("CREATE SCHEMA IF NOT EXISTS docurag")
    op.execute("CREATE SCHEMA IF NOT EXISTS audit")

    # ── Enable extensions ─────────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # ── Enum types ────────────────────────────────────────────────────────────
    document_type_enum = postgresql.ENUM(
        "pdf", "docx", "xlsx", "csv", "image", "unknown",
        name="document_type", schema="docurag", create_type=True,
    )
    document_type_enum.create(op.get_bind(), checkfirst=True)

    document_status_enum = postgresql.ENUM(
        "pending", "classifying", "classified", "extracting", "extracted",
        "chunking", "chunked", "embedding", "ready", "failed",
        name="document_status", schema="docurag", create_type=True,
    )
    document_status_enum.create(op.get_bind(), checkfirst=True)

    page_type_enum = postgresql.ENUM(
        "digital", "scanned", "mixed", "unknown",
        name="page_type", schema="docurag", create_type=True,
    )
    page_type_enum.create(op.get_bind(), checkfirst=True)

    job_status_enum = postgresql.ENUM(
        "queued", "running", "success", "failed", "retrying",
        name="job_status", schema="docurag", create_type=True,
    )
    job_status_enum.create(op.get_bind(), checkfirst=True)

    chunk_type_enum = postgresql.ENUM(
        "text", "table", "heading", "caption", "formula", "code", "list", "footer", "header",
        name="chunk_type", schema="docurag", create_type=True,
    )
    chunk_type_enum.create(op.get_bind(), checkfirst=True)

    # ── documents table ───────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()"),
                  comment="Unique document identifier, used in all citations"),
        sa.Column("original_filename", sa.String(512), nullable=False,
                  comment="Original filename as uploaded by the user"),
        sa.Column("stored_filename", sa.String(512), nullable=False,
                  comment="Filename in raw storage directory (uuid-based)"),
        sa.Column("file_path", sa.Text, nullable=False,
                  comment="Absolute path to the raw stored file"),
        sa.Column("file_type", sa.Enum(name="document_type", schema="docurag"), nullable=False,
                  comment="Detected document type"),
        sa.Column("mime_type", sa.String(128), nullable=False,
                  comment="MIME type detected at ingestion"),
        sa.Column("file_size_bytes", sa.BigInteger, nullable=False,
                  comment="File size in bytes"),
        sa.Column("checksum", sa.String(64), nullable=False,
                  comment="SHA-256 hex digest for deduplication"),
        sa.Column("title", sa.String(512), nullable=True,
                  comment="Extracted or user-provided document title"),
        sa.Column("detected_language", sa.String(16), nullable=True,
                  comment="BCP-47 language code"),
        sa.Column("language_confidence", sa.Float, nullable=True,
                  comment="Language detection confidence (0-1)"),
        sa.Column("page_count", sa.Integer, nullable=True,
                  comment="Total pages in document"),
        sa.Column("version", sa.String(32), nullable=False, server_default="1",
                  comment="Document version"),
        sa.Column("status", sa.Enum(name="document_status", schema="docurag"), nullable=False,
                  server_default="pending", comment="Pipeline processing status"),
        sa.Column("error_message", sa.Text, nullable=True,
                  comment="Error details if status=failed"),
        sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("extra_metadata", postgresql.JSONB, nullable=True,
                  comment="Arbitrary additional metadata"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
        schema="docurag",
    )
    op.create_unique_constraint("uq_document_checksum", "documents", ["checksum"], schema="docurag")
    op.create_index("ix_documents_status", "documents", ["status"], schema="docurag")
    op.create_index("ix_documents_file_type", "documents", ["file_type"], schema="docurag")
    op.create_index("ix_documents_created_at", "documents", ["created_at"], schema="docurag")

    # Auto-update updated_at on row change
    op.execute("""
        CREATE OR REPLACE FUNCTION docurag.update_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_documents_updated_at
        BEFORE UPDATE ON docurag.documents
        FOR EACH ROW EXECUTE FUNCTION docurag.update_updated_at();
    """)

    # ── pages table ───────────────────────────────────────────────────────────
    op.create_table(
        "pages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False,
                  comment="Parent document"),
        sa.Column("page_number", sa.Integer, nullable=False,
                  comment="1-based page number"),
        sa.Column("page_type", sa.Enum(name="page_type", schema="docurag"), nullable=False,
                  server_default="unknown", comment="digital / scanned / mixed / unknown"),
        sa.Column("raw_text", sa.Text, nullable=True,
                  comment="Raw text extracted from this page"),
        sa.Column("ocr_applied", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("ocr_confidence", sa.Float, nullable=True),
        sa.Column("ocr_engine", sa.String(64), nullable=True),
        sa.Column("width", sa.Float, nullable=True),
        sa.Column("height", sa.Float, nullable=True),
        sa.Column("layout_metadata", postgresql.JSONB, nullable=True,
                  comment="Detected layout regions"),
        sa.Column("detected_language", sa.String(16), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(
            ["document_id"], ["docurag.documents.id"],
            ondelete="CASCADE", name="fk_pages_document_id",
        ),
        schema="docurag",
    )
    op.create_index("ix_pages_document_id", "pages", ["document_id"], schema="docurag")
    op.create_index("ix_pages_page_type", "pages", ["page_type"], schema="docurag")
    op.create_index("ix_pages_doc_page", "pages", ["document_id", "page_number"], schema="docurag")

    # ── chunks table ──────────────────────────────────────────────────────────
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("page_number", sa.Integer, nullable=True,
                  comment="Source page number (1-based) for citation"),
        sa.Column("section_title", sa.String(512), nullable=True,
                  comment="Nearest heading for citation"),
        sa.Column("chunk_index", sa.Integer, nullable=False, server_default="0",
                  comment="Ordinal within the page"),
        sa.Column("text", sa.Text, nullable=False, comment="Chunk text content"),
        sa.Column("chunk_type", sa.Enum(name="chunk_type", schema="docurag"), nullable=False,
                  server_default="text"),
        sa.Column("token_count", sa.Integer, nullable=True,
                  comment="Approximate token count"),
        # Vector embedding column — 384 dims for all-MiniLM-L6-v2
        sa.Column("embedding", Vector(384), nullable=True,
                  comment="Sentence embedding vector (pgvector)"),
        sa.Column("embedding_model", sa.String(256), nullable=True),
        sa.Column("embedding_generated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("extra_metadata", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(
            ["document_id"], ["docurag.documents.id"],
            ondelete="CASCADE", name="fk_chunks_document_id",
        ),
        sa.ForeignKeyConstraint(
            ["page_id"], ["docurag.pages.id"],
            ondelete="SET NULL", name="fk_chunks_page_id",
        ),
        schema="docurag",
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"], schema="docurag")
    op.create_index("ix_chunks_page_id", "chunks", ["page_id"], schema="docurag")
    op.create_index("ix_chunks_chunk_type", "chunks", ["chunk_type"], schema="docurag")

    # IVFFlat vector index — tune lists= based on dataset size
    op.execute("""
        CREATE INDEX ix_chunks_embedding_ivfflat
        ON docurag.chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        WHERE embedding IS NOT NULL;
    """)

    # ── processing_jobs table ─────────────────────────────────────────────────
    op.create_table(
        "processing_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("celery_task_id", sa.String(256), nullable=True,
                  comment="Celery task UUID for status polling"),
        sa.Column("job_type", sa.String(64), nullable=False,
                  comment="Pipeline stage name"),
        sa.Column("status", sa.Enum(name="job_status", schema="docurag"), nullable=False,
                  server_default="queued"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("retry_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("result_metadata", postgresql.JSONB, nullable=True,
                  comment="Stage-specific output summary"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(
            ["document_id"], ["docurag.documents.id"],
            ondelete="CASCADE", name="fk_jobs_document_id",
        ),
        schema="docurag",
    )
    op.create_index("ix_jobs_document_id", "processing_jobs", ["document_id"], schema="docurag")
    op.create_index("ix_jobs_status", "processing_jobs", ["status"], schema="docurag")
    op.create_index("ix_jobs_celery_task_id", "processing_jobs", ["celery_task_id"], schema="docurag")


def downgrade() -> None:
    """Drop all tables and types in reverse order."""
    op.execute("DROP TABLE IF EXISTS docurag.processing_jobs CASCADE")
    op.execute("DROP TABLE IF EXISTS docurag.chunks CASCADE")
    op.execute("DROP TABLE IF EXISTS docurag.pages CASCADE")
    op.execute("DROP TABLE IF EXISTS docurag.documents CASCADE")
    op.execute("DROP FUNCTION IF EXISTS docurag.update_updated_at() CASCADE")
    op.execute("DROP TYPE IF EXISTS docurag.chunk_type CASCADE")
    op.execute("DROP TYPE IF EXISTS docurag.job_status CASCADE")
    op.execute("DROP TYPE IF EXISTS docurag.page_type CASCADE")
    op.execute("DROP TYPE IF EXISTS docurag.document_status CASCADE")
    op.execute("DROP TYPE IF EXISTS docurag.document_type CASCADE")
    op.execute("DROP SCHEMA IF EXISTS docurag CASCADE")
    op.execute("DROP SCHEMA IF EXISTS audit CASCADE")
