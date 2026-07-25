"""DocuRAG — Shared Enumeration Types

All enums are defined here to prevent circular imports and serve as the
single source of truth for domain vocabulary used across models, schemas,
and pipeline logic.
"""
from __future__ import annotations

import enum


class DocumentType(str, enum.Enum):
    """Supported input document formats."""

    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"
    IMAGE = "image"  # PNG, JPG, TIFF, BMP
    UNKNOWN = "unknown"


class DocumentStatus(str, enum.Enum):
    """
    Document lifecycle states.

    Transitions:
        PENDING → CLASSIFYING → EXTRACTING → EMBEDDING → READY
                                 → FAILED (from any state)
    """

    PENDING = "pending"             # Uploaded, awaiting pipeline
    CLASSIFYING = "classifying"     # Page-level classification in progress
    CLASSIFIED = "classified"       # Classification complete
    EXTRACTING = "extracting"       # Text/table/OCR extraction in progress
    EXTRACTED = "extracted"         # Extraction complete
    CHUNKING = "chunking"           # Semantic chunking in progress
    CHUNKED = "chunked"             # Chunking complete
    EMBEDDING = "embedding"         # Embedding generation in progress
    READY = "ready"                 # Fully processed, queryable
    FAILED = "failed"               # Terminal failure state


class PageType(str, enum.Enum):
    """
    Per-page classification result.

    Determines the extraction route:
    - DIGITAL  → direct text extraction (pdfplumber)
    - SCANNED  → OCR pipeline (Tesseract/EasyOCR)
    - MIXED    → split into digital and scanned regions
    """

    DIGITAL = "digital"   # Native PDF text layer present
    SCANNED = "scanned"   # Image-only, requires OCR
    MIXED = "mixed"       # Partial text + image regions
    UNKNOWN = "unknown"   # Classification failed


class JobStatus(str, enum.Enum):
    """Celery async job lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


class ChunkType(str, enum.Enum):
    """Semantic content type of a document chunk."""

    TEXT = "text"
    TABLE = "table"
    HEADING = "heading"
    CAPTION = "caption"
    FORMULA = "formula"
    CODE = "code"
    LIST = "list"
    FOOTER = "footer"
    HEADER = "header"
