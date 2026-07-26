"""
DocuRAG — Test Suite: Document Ingestion Pipeline

Tests the core ingestion pipeline including:
- File validation (size, extension)
- SHA-256 checksum computation
- Page classification (digital, scanned, mixed)
- Document and page record creation
- Duplicate detection

All tests use pytest-asyncio and mock DB sessions to remain unit-testable
without requiring a live PostgreSQL connection.
"""
from __future__ import annotations

import hashlib
import io
import tempfile
import uuid
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database_models.shared_enums import DocumentStatus, DocumentType, PageType
from src.document_processing.processing_schemas import PageClassificationResult
from src.document_processing.document_classifier import (
    _determine_page_type,
    classify_image_document,
    classify_text_document,
    _summarise_classification,
)
from src.shared_utilities.file_operations import (
    compute_sha256,
    detect_mime_type,
    generate_stored_filename,
    mime_to_document_type,
    validate_file_extension,
    validate_file_size,
)
from src.shared_utilities.custom_exceptions import (
    DuplicateDocumentError,
    FileSizeLimitError,
    UnsupportedFileTypeError,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_pdf(tmp_path: Path) -> Path:
    """Create a minimal valid PDF-like file for testing."""
    pdf_file = tmp_path / "test_document.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 minimal test content for DocuRAG unit tests")
    return pdf_file


@pytest.fixture
def temp_csv(tmp_path: Path) -> Path:
    """Create a small CSV file for testing."""
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text("name,value,description\nAlpha,1,First item\nBeta,2,Second item\n")
    return csv_file


@pytest.fixture
def temp_large_file(tmp_path: Path) -> Path:
    """Create a file that exceeds the size limit."""
    large_file = tmp_path / "large_file.pdf"
    # Write 101 MB
    large_file.write_bytes(b"X" * (101 * 1024 * 1024))
    return large_file


@pytest.fixture
def temp_unsupported_file(tmp_path: Path) -> Path:
    """Create a file with an unsupported extension."""
    unsupported = tmp_path / "malware.exe"
    unsupported.write_bytes(b"MZ\x90\x00 fake executable")
    return unsupported


# ── File Utility Tests ─────────────────────────────────────────────────────────

class TestComputeSha256:
    """Tests for the SHA-256 checksum utility."""

    def test_known_content(self, tmp_path: Path) -> None:
        """Checksum of known content should match expected hash."""
        test_file = tmp_path / "known.txt"
        content = b"hello docurag"
        test_file.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        result = compute_sha256(test_file)
        assert result == expected

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Different content must produce different checksums."""
        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        file_a.write_bytes(b"content A")
        file_b.write_bytes(b"content B")

        assert compute_sha256(file_a) != compute_sha256(file_b)

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        """Identical content must produce identical checksums (deduplication)."""
        file_a = tmp_path / "a.pdf"
        file_b = tmp_path / "b.pdf"
        file_a.write_bytes(b"identical content")
        file_b.write_bytes(b"identical content")

        assert compute_sha256(file_a) == compute_sha256(file_b)

    def test_returns_lowercase_hex(self, tmp_path: Path) -> None:
        """Checksum must be a lowercase hex string."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\xff\xab")

        result = compute_sha256(test_file)
        assert result == result.lower()
        assert all(c in "0123456789abcdef" for c in result)
        assert len(result) == 64  # SHA-256 is always 64 hex chars


class TestDetectMimeType:
    """Tests for MIME type detection from file extensions."""

    @pytest.mark.parametrize("filename,expected_mime", [
        ("report.pdf", "application/pdf"),
        ("data.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("sheet.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("records.csv", "text/csv"),
        ("scan.png", "image/png"),
        ("photo.jpg", "image/jpeg"),
        ("photo.jpeg", "image/jpeg"),
        ("doc.tiff", "image/tiff"),
    ])
    def test_known_extensions(self, tmp_path: Path, filename: str, expected_mime: str) -> None:
        """Known extensions should return correct MIME types."""
        test_file = tmp_path / filename
        test_file.write_bytes(b"dummy content")
        assert detect_mime_type(test_file) == expected_mime


class TestMimeToDocumentType:
    """Tests for MIME-type to DocumentType mapping."""

    @pytest.mark.parametrize("mime,expected_type", [
        ("application/pdf", "pdf"),
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
        ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"),
        ("text/csv", "csv"),
        ("image/png", "image"),
        ("image/jpeg", "image"),
        ("image/tiff", "image"),
        ("application/unknown", "unknown"),
    ])
    def test_mapping(self, mime: str, expected_type: str) -> None:
        assert mime_to_document_type(mime) == expected_type


class TestValidateFileExtension:
    """Tests for file extension validation."""

    def test_valid_pdf_extension(self, temp_pdf: Path) -> None:
        """No exception for supported extensions."""
        validate_file_extension(temp_pdf)  # Should not raise

    @pytest.mark.parametrize("ext", [".pdf", ".docx", ".xlsx", ".csv", ".png", ".jpg", ".jpeg"])
    def test_all_valid_extensions(self, tmp_path: Path, ext: str) -> None:
        f = tmp_path / f"test{ext}"
        f.write_bytes(b"content")
        validate_file_extension(f)  # Should not raise

    def test_invalid_extension_raises(self, temp_unsupported_file: Path) -> None:
        """Unsupported extension must raise ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            validate_file_extension(temp_unsupported_file)


class TestValidateFileSize:
    """Tests for file size validation."""

    def test_small_file_passes(self, temp_pdf: Path) -> None:
        """Small files under the limit should pass."""
        validate_file_size(temp_pdf, max_bytes=100 * 1024 * 1024)

    def test_oversized_file_raises(self, temp_large_file: Path) -> None:
        """Files exceeding the limit must raise ValueError."""
        with pytest.raises(ValueError, match="exceeding the"):
            validate_file_size(temp_large_file, max_bytes=100 * 1024 * 1024)


class TestGenerateStoredFilename:
    """Tests for unique stored filename generation."""

    def test_preserves_extension(self) -> None:
        """Stored filename must preserve the original extension."""
        result = generate_stored_filename("document.pdf")
        assert result.endswith(".pdf")

    def test_generates_unique_names(self) -> None:
        """Each call must produce a unique filename."""
        names = {generate_stored_filename("test.pdf") for _ in range(100)}
        assert len(names) == 100

    def test_extension_lowercased(self) -> None:
        """Extension should be lowercase regardless of input case."""
        result = generate_stored_filename("REPORT.PDF")
        assert result.endswith(".pdf")


# ── Classifier Tests ──────────────────────────────────────────────────────────

class TestDeterminePageType:
    """Tests for the page type determination heuristic."""

    @pytest.mark.parametrize("text_count,has_images,threshold,expected", [
        (500, False, 50, PageType.DIGITAL),    # Rich text, no images
        (500, True, 50, PageType.MIXED),       # Rich text + images
        (10, True, 50, PageType.SCANNED),      # Sparse text + images = scanned
        (0, True, 50, PageType.SCANNED),       # No text + images = scanned
        (0, False, 50, PageType.SCANNED),      # No text, no images = treat as scanned
        (50, False, 50, PageType.DIGITAL),     # Exactly at threshold = digital
        (49, False, 50, PageType.SCANNED),     # Just below threshold = scanned
    ])
    def test_heuristic(
        self,
        text_count: int,
        has_images: bool,
        threshold: int,
        expected: PageType,
    ) -> None:
        result = _determine_page_type(text_count, has_images, threshold)
        assert result == expected


class TestClassifyImageDocument:
    """Tests for standalone image file classification."""

    def test_returns_single_scanned_page(self) -> None:
        results = classify_image_document()
        assert len(results) == 1
        assert results[0].page_type == PageType.SCANNED
        assert results[0].page_number == 1
        assert results[0].has_images is True
        assert results[0].text_char_count == 0


class TestClassifyTextDocument:
    """Tests for flat text/CSV document classification."""

    def test_returns_single_digital_page(self) -> None:
        text = "name,value\nAlpha,1\nBeta,2\n"
        results = classify_text_document(text)
        assert len(results) == 1
        assert results[0].page_type == PageType.DIGITAL
        assert results[0].text_char_count == len(text)
        assert results[0].has_images is False

    def test_empty_text(self) -> None:
        results = classify_text_document("")
        assert results[0].page_type == PageType.DIGITAL
        assert results[0].text_char_count == 0


class TestSummariseClassification:
    """Tests for classification result summarisation."""

    def test_mixed_page_types(self) -> None:
        results = [
            PageClassificationResult(page_number=1, page_type=PageType.DIGITAL, text_char_count=100, has_images=False),
            PageClassificationResult(page_number=2, page_type=PageType.SCANNED, text_char_count=0, has_images=True),
            PageClassificationResult(page_number=3, page_type=PageType.MIXED, text_char_count=200, has_images=True),
            PageClassificationResult(page_number=4, page_type=PageType.DIGITAL, text_char_count=300, has_images=False),
        ]
        summary = _summarise_classification(results)
        assert summary["digital_pages"] == 2
        assert summary["scanned_pages"] == 1
        assert summary["mixed_pages"] == 1

    def test_all_digital(self) -> None:
        results = [
            PageClassificationResult(page_number=i, page_type=PageType.DIGITAL, text_char_count=100, has_images=False)
            for i in range(1, 6)
        ]
        summary = _summarise_classification(results)
        assert summary["digital_pages"] == 5
        assert summary["scanned_pages"] == 0
        assert summary["mixed_pages"] == 0


# ── Custom Exception Tests ────────────────────────────────────────────────────

class TestCustomExceptions:
    """Tests for structured exception creation and string representation."""

    def test_duplicate_document_error(self) -> None:
        exc = DuplicateDocumentError(
            checksum="abc123",
            existing_doc_id="doc-uuid-here",
        )
        assert "abc123" in str(exc)
        assert exc.context["checksum"] == "abc123"
        assert exc.existing_doc_id == "doc-uuid-here"
        assert exc.checksum == "abc123"

    def test_file_size_limit_error(self) -> None:
        exc = FileSizeLimitError(
            filename="big.pdf",
            size_bytes=200 * 1024 * 1024,
            limit_bytes=100 * 1024 * 1024,
        )
        assert "big.pdf" in str(exc)
        assert exc.context["filename"] == "big.pdf"

    def test_unsupported_file_type_error(self) -> None:
        exc = UnsupportedFileTypeError(file_type=".exe", filename="malware.exe")
        assert ".exe" in str(exc)
        assert "malware.exe" in str(exc)
