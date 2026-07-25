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
    JSON,
)
from sqlalchemy import Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database_models.database_connection import Base
from src.database_models.shared_enums import ChunkType


class Chunk(Base):
    """
    Atomic text segment ready for embedding and retrieval.

    Chunk boundaries are determined by the semantic chunking agent (Phase 3).
    For Phase 0–1, chunks are created as page-level placeholders.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_page_id", "page_id"),
        Index("ix_chunks_chunk_type", "chunk_type"),
        
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    page_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid,
        ForeignKey("pages.id", ondelete="SET NULL"),
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
        Enum(ChunkType, name="chunk_type"),
        nullable=False,
        default=ChunkType.TEXT,
    )
    token_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Approximate token count for context window management",
    )

    # Embedding metadata (vector stored via raw DDL in migration)
    embedding_model: Mapped[Optional[str]] = mapped_column(
        nullable=True,
        comment="Name of the embedding model used",
    )
    embedding_generated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # Use JSON for SQLite compatibility (instead of pgvector Vector)
    embedding = mapped_column(JSON().with_variant(JSON(), "sqlite"), nullable=True)

    # Flexible extra metadata
    extra_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
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
