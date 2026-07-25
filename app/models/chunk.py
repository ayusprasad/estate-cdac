"""
DocuRAG — Chunk ORM Model

Represents a semantically coherent text segment extracted from a page.
Chunks are the atomic unit used for embedding generation and retrieval.

Citation chain: Chunk → Page → Document
  - chunk.document_id / chunk.page_id / chunk.page_number / chunk.section_title
  together form a complete citation reference.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database import Base
from app.models.enums import ChunkType


class Chunk(Base):
    """
    Atomic text segment ready for embedding and retrieval.

    Chunk boundaries are determined by the semantic chunking agent (Phase 3).
    For Phase 0-1, chunks are created as page-level placeholders.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_page_id", "page_id"),
        Index("ix_chunks_chunk_type", "chunk_type"),
        {"schema": "docurag"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("docurag.documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    page_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("docurag.pages.id", ondelete="SET NULL"),
        nullable=True,
        comment="Source page (null for non-paginated documents)",
    )

    # Citation fields — populated at chunk creation
    page_number: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Source page number (1-based) for citation",
    )
    section_title: Mapped[Optional[str]] = mapped_column(
        nullable=True,
        comment="Nearest heading/section title for citation",
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Ordinal index of this chunk within the page",
    )

    # Content
    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="The chunk text content",
    )
    chunk_type: Mapped[ChunkType] = mapped_column(
        Enum(ChunkType, name="chunk_type", schema="docurag"),
        nullable=False,
        default=ChunkType.TEXT,
    )
    token_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Approximate token count for context window management",
    )

    # Embedding (populated in Phase 3)
    # Stored as pgvector VECTOR type via raw DDL in migration
    embedding_model: Mapped[Optional[str]] = mapped_column(
        nullable=True,
        comment="Name of the embedding model used",
    )
    embedding_generated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Flexible extra metadata
    extra_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Table headers, formula latex, image alt-text, etc.",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    page: Mapped[Optional["Page"]] = relationship("Page", back_populates="chunks")

    def __repr__(self) -> str:
        return (
            f"<Chunk id={self.id} page_num={self.page_number} "
            f"type={self.chunk_type.value!r} tokens={self.token_count}>"
        )
