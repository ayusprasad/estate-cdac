"""
DocuRAG — File Utility Functions

Reusable helpers for:
- File type detection (MIME type + extension)
- SHA-256 checksum computation
- Secure file storage (UUID-based naming)
- File size validation
- Temporary file management
"""
from __future__ import annotations

import hashlib
import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Optional

import aiofiles

from application_configuration.logger_setup import get_logger
from application_configuration.environment_settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

_MIME_TO_DOC_TYPE: dict[str, str] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xlsx",
    "text/csv": "csv",
    "text/plain": "csv",
    "image/png": "image",
    "image/jpeg": "image",
    "image/tiff": "image",
    "image/bmp": "image",
    "image/webp": "image",
}

_EXT_TO_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".csv": "text/csv",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
}


def detect_mime_type(file_path: Path) -> str:
    """Detect MIME type using file extension."""
    ext = file_path.suffix.lower()
    if ext in _EXT_TO_MIME:
        return _EXT_TO_MIME[ext]
    mime, _ = mimetypes.guess_type(str(file_path))
    return mime or "application/octet-stream"


def mime_to_document_type(mime_type: str) -> str:
    """Map MIME type string to DocuRAG DocumentType string."""
    return _MIME_TO_DOC_TYPE.get(mime_type, "unknown")


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """
    Compute SHA-256 hex digest of a file by streaming in chunks.

    Args:
        file_path: Path to the file.
        chunk_size: Read buffer size in bytes.

    Returns:
        Lowercase hex SHA-256 digest string (64 characters).
    """
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


async def compute_sha256_async(file_path: Path, chunk_size: int = 65536) -> str:
    """Async variant of compute_sha256."""
    sha256 = hashlib.sha256()
    async with aiofiles.open(file_path, "rb") as f:
        while chunk := await f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_file_size(file_path: Path, max_bytes: Optional[int] = None) -> None:
    """
    Raise ValueError if file exceeds the maximum allowed size.

    Args:
        file_path: Path to the file.
        max_bytes: Maximum size in bytes. Defaults to settings value.
    """
    limit = max_bytes if max_bytes is not None else settings.document.max_file_size_bytes
    size = file_path.stat().st_size
    if size > limit:
        raise ValueError(
            f"File '{file_path.name}' is {size:,} bytes, "
            f"exceeding the {limit:,} byte limit."
        )


def validate_file_extension(file_path: Path) -> None:
    """
    Raise ValueError if file extension is not in the supported list.

    Args:
        file_path: Path to the file.
    """
    ext = file_path.suffix.lower()
    if ext not in settings.document.supported_extensions:
        raise ValueError(
            f"File extension '{ext}' is not supported. "
            f"Allowed: {settings.document.supported_extensions}"
        )


def generate_stored_filename(original_filename: str) -> str:
    """
    Generate a UUID-based storage filename preserving the original extension.

    Args:
        original_filename: The user-uploaded filename.

    Returns:
        A unique filename like '3f2a1b...c4.pdf'.
    """
    ext = Path(original_filename).suffix.lower()
    return f"{uuid.uuid4().hex}{ext}"


def save_upload_to_raw_storage(source_path: Path, stored_filename: str) -> Path:
    """
    Copy a file to the raw document storage directory.

    Args:
        source_path: Temporary path of the uploaded file.
        stored_filename: Target filename in raw storage.

    Returns:
        Absolute Path to the stored file.
    """
    raw_dir = settings.storage.raw_docs_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest_path = raw_dir / stored_filename
    shutil.copy2(source_path, dest_path)
    logger.info(
        "Saved file to raw storage",
        source=str(source_path),
        dest=str(dest_path),
        size_bytes=dest_path.stat().st_size,
    )
    return dest_path


async def save_upload_bytes_async(file_bytes: bytes, stored_filename: str) -> Path:
    """
    Write uploaded bytes directly to raw storage asynchronously.

    Args:
        file_bytes: Raw bytes from the upload.
        stored_filename: Target filename in raw storage.

    Returns:
        Absolute Path to the stored file.
    """
    raw_dir = settings.storage.raw_docs_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest_path = raw_dir / stored_filename
    async with aiofiles.open(dest_path, "wb") as f:
        await f.write(file_bytes)
    logger.info(
        "Saved upload bytes to raw storage",
        dest=str(dest_path),
        size_bytes=len(file_bytes),
    )
    return dest_path


def get_processed_path(document_id: str, filename: str) -> Path:
    """
    Compute the path for a processed file associated with a document.

    Args:
        document_id: UUID string of the parent document.
        filename: Target filename within the processed directory.

    Returns:
        Absolute Path (directory created if needed).
    """
    processed_dir = settings.storage.processed_docs_dir / document_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir / filename
