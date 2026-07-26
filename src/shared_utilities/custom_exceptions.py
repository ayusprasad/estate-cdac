"""
DocuRAG — Custom Exception Hierarchy

All domain-specific exceptions inherit from DocuRAGException.
HTTP-layer exceptions are handled in FastAPI exception handlers.
Pipeline exceptions carry structured context for logging and debugging.
"""
from __future__ import annotations

from typing import Any, Optional


class DocuRAGException(Exception):
    """Base exception for all DocuRAG errors."""

    def __init__(self, message: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{ctx_str}]"
        return self.message


# ── Ingestion Exceptions ──────────────────────────────────────────────────────

class IngestionError(DocuRAGException):
    """Raised when the ingestion pipeline encounters a fatal error."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when an unsupported file type is uploaded."""

    def __init__(self, file_type: str, filename: str) -> None:
        super().__init__(
            f"Unsupported file type '{file_type}' for file '{filename}'.",
            context={"file_type": file_type, "filename": filename},
        )
        self.file_type = file_type
        self.filename = filename


class FileSizeLimitError(IngestionError):
    """Raised when an uploaded file exceeds the size limit."""

    def __init__(self, filename: str, size_bytes: int, limit_bytes: int) -> None:
        super().__init__(
            f"File '{filename}' ({size_bytes:,} bytes) exceeds limit ({limit_bytes:,} bytes).",
            context={"filename": filename, "size_bytes": size_bytes, "limit_bytes": limit_bytes},
        )
        self.filename = filename
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes


class DuplicateDocumentError(IngestionError):
    """Raised when a document with the same checksum already exists."""

    def __init__(self, checksum: str, existing_doc_id: str) -> None:
        super().__init__(
            f"Document with checksum '{checksum}' already exists (id={existing_doc_id}).",
            context={"checksum": checksum, "existing_doc_id": str(existing_doc_id)},
        )
        self.checksum = checksum
        self.existing_doc_id = str(existing_doc_id)


# ── Classification Exceptions ─────────────────────────────────────────────────

class ClassificationError(DocuRAGException):
    """Raised when page classification fails."""


# ── Storage Exceptions ────────────────────────────────────────────────────────

class StorageError(DocuRAGException):
    """Raised when a file storage operation fails."""


# ── Database Exceptions ───────────────────────────────────────────────────────

class DatabaseError(DocuRAGException):
    """Raised when a database operation fails unexpectedly."""


class DocumentNotFoundError(DatabaseError):
    """Raised when a requested document does not exist."""

    def __init__(self, document_id: str) -> None:
        super().__init__(
            f"Document with id='{document_id}' not found.",
            context={"document_id": str(document_id)},
        )
        self.document_id = str(document_id)


# ── Processing Exceptions ─────────────────────────────────────────────────────

class ExtractionError(DocuRAGException):
    """Raised when text or table extraction fails."""


class OCRError(DocuRAGException):
    """Raised when OCR processing fails."""


class EmbeddingError(DocuRAGException):
    """Raised when embedding generation fails."""


class ConfigurationError(DocuRAGException):
    """Raised when required configuration is missing or invalid."""
