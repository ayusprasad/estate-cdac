"""
DocuRAG — Ingestion Pydantic Schemas

Request/response models for the document ingestion API layer.
Separate from ORM models to allow independent evolution of API contracts.

All response schemas include the document_id so callers can poll
for processing status after an async upload.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import DocumentStatus, DocumentType, PageType


# ── Request schemas ─────────────────────────────────────────

class DocumentUploadMetadata(BaseModel):
    """
    Optional caller-supplied metadata attached at upload time.

    These are stored in Document.extra_metadata and can be used for
    filtering, labelling, and citation display.
    """

    model_config = ConfigDict(extra="allow")

    title: Optional[str] = Field(None, description="Human-readable document title")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    source: Optional[str] = Field(None, description="Source system or URL")
    author: Optional[str] = Field(None, description="Document author")
    language_hint: Optional[str] = Field(
        None,
        description="BCP-47 language hint (skips auto-detection if supplied)",
    )


# ── Response schemas ────────────────────────────────────────

class DocumentResponse(BaseModel):
    """Full document representation returned by GET /documents/{id}."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_filename: str
    file_type: DocumentType
    mime_type: str
    file_size_bytes: int
    checksum: str
    title: Optional[str] = None
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None
    page_count: Optional[int] = None
    version: str
    status: DocumentStatus
    error_message: Optional[str] = None
    extra_metadata: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None


class DocumentUploadResponse(BaseModel):
    """Response returned immediately after a successful upload."""

    document_id: uuid.UUID = Field(..., description="Unique ID for this document")
    original_filename: str
    file_type: DocumentType
    file_size_bytes: int
    checksum: str
    status: DocumentStatus
    message: str = "Document accepted for processing"


class PageClassificationResult(BaseModel):
    """Classification result for a single document page."""

    page_number: int = Field(..., ge=1)
    page_type: PageType
    text_char_count: int = Field(..., ge=0, description="Characters extracted from digital layer")
    has_images: bool = False
    width: Optional[float] = None
    height: Optional[float] = None


class DocumentClassificationResponse(BaseModel):
    """Full classification result for a document."""

    document_id: uuid.UUID
    total_pages: int
    digital_pages: int
    scanned_pages: int
    mixed_pages: int
    page_results: list[PageClassificationResult]


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    total: int
    page: int
    page_size: int
    items: list[DocumentResponse]


class ProcessingStatusResponse(BaseModel):
    """Current processing status of a document."""

    document_id: uuid.UUID
    status: DocumentStatus
    error_message: Optional[str] = None
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
