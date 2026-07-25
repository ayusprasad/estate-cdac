"""
DocuRAG — Document ORM Model

Represents a single ingested document. This is the root entity for all
downstream processing — every page, chunk, and embedding traces back to
a Document record via document_id.

Schema fields are chosen to support:
- Traceability (checksum, original_filename, file_path)
- Multi-format support (file_type, mime_type)
- Multi-language support (detected_language)
- Pipeline state management (status, error_message)
- Citation generation (document_id + page number + section)
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    Float,
    Index,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database_models.database_connection import Base
from src.database_models.shared_enums import DocumentStatus, DocumentType


class Document(Base):
    """
    Core document entity.

    Lifecycle:
        PENDING → CLASSIFYING → EXTRACTING → EMBEDDING → READY
                                              ↙↗
                                           FAILED
    """

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("checksum", name="uq_document_checksum"),
        Index("ix_documents_status", "status"),
        Index("ix_documents_file_type", "file_type"),
        Index("ix_documents_created_at", "created_at"),
        
    )

    # ── Primary key ──────────────────────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique document identifier, used in all citations",
    )

    # ── File identity ─────────────────────────────────────────────────────────
    original_filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Original filename as uploaded by the user",
    )
    stored_filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Filename used in the raw storage directory (uuid-based)",
    )
    file_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Absolute path to the raw stored file",
    )
    file_type: Mapped[DocumentType] = mapped_column(
        Enum(DocumentType, name="document_type"),
        nullable=False,
        comment="Detected document type (pdf, docx, xlsx, csv, image)",
    )
    mime_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="MIME type detected at ingestion",
    )
    file_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="File size in bytes",
    )
    checksum: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="SHA-256 hex digest for deduplication and integrity verification",
    )

    # ── Content metadata ──────────────────────────────────────────────────────
    title: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
        comment="Extracted or user-provided document title",
    )
    detected_language: Mapped[Optional[str]] = mapped_column(
        String(16),
        nullable=True,
        comment="BCP-47 language code detected by langdetect",
    )
    language_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Confidence score of language detection (0-1)",
    )
    page_count: Mapped[Optional[int]] = mapped_column(
        nullable=True,
        comment="Total number of pages (for paginated document types)",
    )
    version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="1",
        comment="Document version (incremented on re-upload of same file)",
    )

    # ── Processing state ──────────────────────────────────────────────────────
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status"),
        nullable=False,
        default=DocumentStatus.PENDING,
        comment="Current pipeline processing status",
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error details if status=FAILED",
    )
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    processing_completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ── Extra metadata (flexible JSON bag) ────────────────────────────────────
    extra_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Arbitrary additional metadata (author, tags, source, etc.)",
    )

    # ── Audit timestamps ──────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    pages: Mapped[list["Page"]] = relationship(
        "Page",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="select",
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="select",
    )
    processing_jobs: Mapped[list["ProcessingJob"]] = relationship(
        "ProcessingJob",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:
        return (
            f"<Document id={self.id} filename={self.original_filename!r} "
            f"status={self.status.value!r}>"
        )
