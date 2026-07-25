"""
DocuRAG — Page ORM Model

Represents a single page within an ingested document.
Each page stores its classification result (digital/scanned/mixed),
raw extracted text, and structural metadata needed for citation generation.

Every downstream chunk and embedding can be traced back to a specific
page through: Chunk.page_id → Page.id → Page.document_id → Document.id
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
    JSON,
    Text,
    func,
)
from sqlalchemy import Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database_models.database_connection import Base
from src.database_models.shared_enums import PageType


class Page(Base):
    """
    Represents a single page of an ingested document.

    For non-paginated formats (CSV, images), a single Page record
    with page_number=1 is created to maintain a consistent data model.
    """

    __tablename__ = "pages"
    __table_args__ = (
        Index("ix_pages_document_id", "document_id"),
        Index("ix_pages_page_type", "page_type"),
        Index("ix_pages_doc_page", "document_id", "page_number"),
        
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )

    # Foreign key to parent document
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="1-based page number within the document",
    )

    page_type: Mapped[PageType] = mapped_column(
        Enum(PageType, name="page_type"),
        nullable=False,
        default=PageType.UNKNOWN,
        comment="Classification: digital, scanned, mixed, or unknown",
    )

    # Raw extracted text (from digital layer or OCR)
    raw_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Raw text extracted from this page (pre-chunking)",
    )

    # OCR-specific metadata
    ocr_applied: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
        comment="Whether OCR was run on this page",
    )
    ocr_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Mean OCR confidence score (0–100)",
    )
    ocr_engine: Mapped[Optional[str]] = mapped_column(
        nullable=True,
        comment="OCR engine used (tesseract, easyocr)",
    )

    # Page dimensions (pixels for scanned, points for digital PDF)
    width: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Structural layout metadata (JSON bag for extensibility)
    layout_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Detected layout regions, headings, table locations, etc.",
    )

    # Detected language override (if different from parent document)
    detected_language: Mapped[Optional[str]] = mapped_column(
        nullable=True,
        comment="BCP-47 language code for this page (if different from document)",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="pages")
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="page",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<Page id={self.id} doc={self.document_id} "
            f"page_num={self.page_number} type={self.page_type.value!r}>"
        )
