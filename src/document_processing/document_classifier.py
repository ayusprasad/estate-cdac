"""
DocuRAG — Document Page Classifier

Determines the type of each page in a document:
  - DIGITAL : Native PDF text layer with sufficient extractable text
  - SCANNED : Image-only or near-zero text — requires OCR
  - MIXED   : Partial text + significant image regions

Classification is intentionally conservative: any ambiguous page is
classified as MIXED to ensure OCR is applied where needed.

Design:
- Pure function interface for testability
- No I/O side effects — callers manage file handles
- Threshold-based heuristic (char count per page) plus image presence check
- Returns structured PageClassificationResult objects for direct DB storage

Extensibility:
- Replace or supplement the heuristic with an ML classifier in Phase 2
  by swapping the _classify_single_page() implementation
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from application_configuration.logger_setup import get_logger
from application_configuration.environment_settings import get_settings
from src.document_processing.processing_schemas import PageClassificationResult
from src.database_models.shared_enums import PageType

logger = get_logger(__name__)
settings = get_settings()

# Minimum text characters on a page to consider it digital (not scanned)
_DIGITAL_TEXT_THRESHOLD = settings.document.scanned_page_text_threshold

# If >N% of page area is covered by images AND text < threshold, classify as MIXED
_MIXED_IMAGE_COVERAGE_RATIO = 0.30


def classify_pdf_pages(pdf_path: Path) -> list[PageClassificationResult]:
    """
    Classify every page of a PDF as digital, scanned, or mixed.

    This is the primary entry point for PDF classification.
    Uses pdfplumber for text extraction and PyMuPDF for image detection.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        List of PageClassificationResult, one per page (1-indexed).

    Raises:
        ClassificationError: If the PDF cannot be opened or parsed.
    """
    results: list[PageClassificationResult] = []

    try:
        import pdfplumber  # type: ignore[import]
        import fitz  # PyMuPDF  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pdfplumber and PyMuPDF are required for PDF classification. "
            "Install with: pip install pdfplumber PyMuPDF"
        ) from exc

    logger.info("Starting PDF page classification", pdf_path=str(pdf_path))

    try:
        # Open with pdfplumber for text extraction
        with pdfplumber.open(pdf_path) as pdf:
            # Open with PyMuPDF for image detection
            fitz_doc = fitz.open(str(pdf_path))

            for page_index, plumber_page in enumerate(pdf.pages):
                page_number = page_index + 1  # 1-based

                result = _classify_single_page(
                    page_number=page_number,
                    plumber_page=plumber_page,
                    fitz_page=fitz_doc[page_index],
                )
                results.append(result)
                logger.debug(
                    "Page classified",
                    page_number=page_number,
                    page_type=result.page_type.value,
                    text_chars=result.text_char_count,
                    has_images=result.has_images,
                )

            fitz_doc.close()

    except Exception as exc:
        from src.shared_utilities.custom_exceptions import ClassificationError
        raise ClassificationError(
            f"Failed to classify PDF pages: {exc}",
            context={"pdf_path": str(pdf_path)},
        ) from exc

    # Log summary
    counts = _summarise_classification(results)
    logger.info(
        "PDF classification complete",
        pdf_path=str(pdf_path),
        total_pages=len(results),
        **counts,
    )

    return results


def _classify_single_page(
    page_number: int,
    plumber_page: object,
    fitz_page: object,
) -> PageClassificationResult:
    """
    Classify a single PDF page using text density and image presence heuristics.

    Args:
        page_number: 1-based page number.
        plumber_page: pdfplumber Page object.
        fitz_page: PyMuPDF Page object.

    Returns:
        PageClassificationResult with classification and metrics.
    """
    # ─ Extract text from digital layer
    try:
        extracted_text: str = plumber_page.extract_text() or ""
    except Exception:  # noqa: BLE001
        extracted_text = ""

    text_char_count = len(extracted_text.strip())

    # ─ Detect images via PyMuPDF
    try:
        image_list = fitz_page.get_images(full=True)
        has_images = len(image_list) > 0
    except Exception:  # noqa: BLE001
        has_images = False
        image_list = []

    # ─ Get page dimensions
    try:
        width = float(plumber_page.width) if hasattr(plumber_page, 'width') else None
        height = float(plumber_page.height) if hasattr(plumber_page, 'height') else None
    except Exception:  # noqa: BLE001
        width = height = None

    # ─ Classify based on heuristics
    page_type = _determine_page_type(
        text_char_count=text_char_count,
        has_images=has_images,
        threshold=_DIGITAL_TEXT_THRESHOLD,
    )

    return PageClassificationResult(
        page_number=page_number,
        page_type=page_type,
        text_char_count=text_char_count,
        has_images=has_images,
        width=width,
        height=height,
    )


def _determine_page_type(
    text_char_count: int,
    has_images: bool,
    threshold: int,
) -> PageType:
    """
    Determine PageType from text character count and image presence.

    Decision table:
    | text >= threshold | has_images | Result  |
    |-------------------|------------|---------|
    | True              | False      | DIGITAL |
    | True              | True       | MIXED   |
    | False             | True       | SCANNED |
    | False             | False      | SCANNED | (blank or broken page)

    Args:
        text_char_count: Number of characters extracted from the digital text layer.
        has_images: Whether the page contains embedded images.
        threshold: Minimum chars to treat page as having meaningful text.

    Returns:
        PageType enum value.
    """
    has_text = text_char_count >= threshold

    if has_text and not has_images:
        return PageType.DIGITAL
    elif has_text and has_images:
        return PageType.MIXED
    elif not has_text and has_images:
        return PageType.SCANNED
    else:
        # No text and no images — likely blank; treat as scanned for safety
        return PageType.SCANNED


def classify_image_document() -> list[PageClassificationResult]:
    """
    Return a single-page classification result for standalone image files.

    Image files (PNG, JPG, TIFF, BMP) always require OCR —
    they are always classified as SCANNED.

    Returns:
        Single-element list with SCANNED classification for page 1.
    """
    return [
        PageClassificationResult(
            page_number=1,
            page_type=PageType.SCANNED,
            text_char_count=0,
            has_images=True,
        )
    ]


def classify_text_document(
    text: str,
    page_number: int = 1,
) -> list[PageClassificationResult]:
    """
    Return a single-page DIGITAL classification for plain-text documents (CSV, TXT).

    Args:
        text: Full document text content.
        page_number: Page number (always 1 for flat documents).

    Returns:
        Single-element list with DIGITAL classification.
    """
    return [
        PageClassificationResult(
            page_number=page_number,
            page_type=PageType.DIGITAL,
            text_char_count=len(text),
            has_images=False,
        )
    ]


def _summarise_classification(
    results: list[PageClassificationResult],
) -> dict[str, int]:
    """Count pages by type from a list of classification results."""
    from collections import Counter
    counts = Counter(r.page_type for r in results)
    return {
        "digital_pages": counts.get(PageType.DIGITAL, 0),
        "scanned_pages": counts.get(PageType.SCANNED, 0),
        "mixed_pages": counts.get(PageType.MIXED, 0),
    }
