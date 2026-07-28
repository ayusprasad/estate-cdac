"""
DocuRAG — Unified Document Parser

Implements multi-format document extraction matching the architecture flowchart:
- PDF (Digital via PyMuPDF / fitz, Tables via pdfplumber, Scanned via Tesseract/OCR)
- Tabular Data (CSV, Excel via Pandas & openpyxl)
- JSON (Structured transaction records)
- Images (PNG, JPG, TIFF via OCR)

Outputs a unified `CanonicalDocument` model.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import fitz  # PyMuPDF
import pandas as pd

from src.document_processing.canonical_json_model import (
    CanonicalDocument,
    CanonicalMetadata,
    CanonicalPage,
    CanonicalSection,
    CanonicalTable,
)
from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file for deduplication."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class UnifiedDocumentParser:
    """Multi-format document parser converting files to CanonicalDocument."""

    def __init__(self, use_ocr: bool = True):
        self.use_ocr = use_ocr

    def parse(self, file_path: Path, document_id: Optional[str] = None) -> CanonicalDocument:
        """Route parsing based on file extension."""
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        doc_id = document_id or str(uuid4())
        checksum = compute_file_sha256(file_path)

        metadata = CanonicalMetadata(
            title=file_path.stem,
            source=str(file_path),
            checksum=checksum,
            file_type=ext.lstrip("."),
            file_size_bytes=file_path.stat().st_size,
        )

        if ext in (".pdf",):
            return self._parse_pdf(file_path, doc_id, metadata)
        elif ext in (".csv", ".xlsx", ".xls"):
            return self._parse_tabular(file_path, doc_id, metadata)
        elif ext in (".json",):
            return self._parse_json(file_path, doc_id, metadata)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
            return self._parse_image(file_path, doc_id, metadata)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _parse_pdf(self, file_path: Path, doc_id: str, metadata: CanonicalMetadata) -> CanonicalDocument:
        """Parse PDF using PyMuPDF (text/layout) and optional pdfplumber (tables)."""
        logger.info("Parsing PDF with PyMuPDF", file_path=str(file_path))
        doc = fitz.open(str(file_path))
        
        canonical_pages: List[CanonicalPage] = []
        canonical_sections: List[CanonicalSection] = []
        canonical_tables: List[CanonicalTable] = []
        full_text_parts: List[str] = []

        # Heading detection pattern: lines starting with #, Section, Chapter, Act, Issue, Clarification, or digits.
        heading_pattern = re.compile(
            r"^(?:#+|section|chapter|act|part|schedule|issue|clarification|\d+[\.\)])\s+.*$",
            re.IGNORECASE,
        )

        for page_idx, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            char_count = len(text.strip())
            page_type = "digital" if char_count >= 50 else "scanned"

            # If page is scanned and OCR is enabled, attempt basic PyMuPDF OCR fallback
            if page_type == "scanned" and self.use_ocr:
                try:
                    tp = page.get_textpage_ocr()
                    ocr_text = tp.extractText() or ""
                    if len(ocr_text.strip()) > char_count:
                        text = ocr_text
                        page_type = "mixed"
                except Exception:
                    pass

            text = text.strip()
            full_text_parts.append(f"--- Page {page_idx} ---\n{text}")

            # Extract headings / sections
            lines = text.split("\n")
            for line in lines:
                clean_line = line.strip()
                if heading_pattern.match(clean_line) and len(clean_line) < 120:
                    canonical_sections.append(
                        CanonicalSection(
                            title=clean_line,
                            level=1,
                            page_number=page_idx,
                            text=clean_line,
                        )
                    )

            canonical_pages.append(
                CanonicalPage(
                    page_number=page_idx,
                    page_type=page_type,
                    text=text,
                    width=page.rect.width,
                    height=page.rect.height,
                )
            )

        doc.close()

        # Optional pdfplumber table extraction if pdfplumber is available
        try:
            import pdfplumber
            with pdfplumber.open(str(file_path)) as pdf_p:
                for page_idx, p_p in enumerate(pdf_p.pages, start=1):
                    tables = p_p.extract_tables() or []
                    for t_idx, table_data in enumerate(tables, start=1):
                        if not table_data or len(table_data) < 2:
                            continue
                        headers = [str(cell or "").strip() for cell in table_data[0]]
                        rows = [[str(cell or "").strip() for cell in row] for row in table_data[1:]]
                        canonical_tables.append(
                            CanonicalTable(
                                page_number=page_idx,
                                caption=f"Table {t_idx} on Page {page_idx}",
                                headers=headers,
                                rows=rows,
                            )
                        )
        except Exception as exc:
            logger.debug("pdfplumber table extraction skipped", error=str(exc))

        full_text = "\n\n".join(full_text_parts)

        return CanonicalDocument(
            document_id=doc_id,
            original_filename=file_path.name,
            metadata=metadata,
            total_pages=len(canonical_pages),
            pages=canonical_pages,
            sections=canonical_sections,
            tables=canonical_tables,
            full_text=full_text,
        )

    def _parse_tabular(self, file_path: Path, doc_id: str, metadata: CanonicalMetadata) -> CanonicalDocument:
        """Parse Excel / CSV using Pandas into CanonicalDocument."""
        logger.info("Parsing tabular file with Pandas", file_path=str(file_path))
        ext = file_path.suffix.lower()

        if ext == ".csv":
            df = pd.read_csv(file_path)
            sheets = {"Sheet1": df}
        else:
            sheets = pd.read_excel(file_path, sheet_name=None)

        canonical_pages: List[CanonicalPage] = []
        canonical_tables: List[CanonicalTable] = []
        full_text_parts: List[str] = []

        page_num = 1
        for sheet_name, df in sheets.items():
            headers = [str(c) for c in df.columns]
            rows = [[str(val) for val in row] for row in df.values]

            text_rep = f"Sheet: {sheet_name}\n" + df.to_string(index=False)
            full_text_parts.append(text_rep)

            table = CanonicalTable(
                page_number=page_num,
                caption=f"Sheet: {sheet_name}",
                headers=headers,
                rows=rows,
            )
            canonical_tables.append(table)

            canonical_pages.append(
                CanonicalPage(
                    page_number=page_num,
                    page_type="digital",
                    text=text_rep,
                    tables=[table],
                )
            )
            page_num += 1

        return CanonicalDocument(
            document_id=doc_id,
            original_filename=file_path.name,
            metadata=metadata,
            total_pages=len(canonical_pages),
            pages=canonical_pages,
            tables=canonical_tables,
            full_text="\n\n".join(full_text_parts),
        )

    def _parse_json(self, file_path: Path, doc_id: str, metadata: CanonicalMetadata) -> CanonicalDocument:
        """Parse JSON transaction records or documents."""
        logger.info("Parsing JSON file", file_path=str(file_path))
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        text_content = json.dumps(data, indent=2)

        page = CanonicalPage(
            page_number=1,
            page_type="digital",
            text=text_content,
        )

        return CanonicalDocument(
            document_id=doc_id,
            original_filename=file_path.name,
            metadata=metadata,
            total_pages=1,
            pages=[page],
            full_text=text_content,
        )

    def _parse_image(self, file_path: Path, doc_id: str, metadata: CanonicalMetadata) -> CanonicalDocument:
        """Parse standalone image file using pytesseract OCR if available."""
        logger.info("Parsing image file with OCR", file_path=str(file_path))
        text = ""
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img) or ""
        except Exception as exc:
            logger.warning("pytesseract OCR unavailable or failed", error=str(exc))
            text = f"[Image file: {file_path.name}]"

        page = CanonicalPage(
            page_number=1,
            page_type="scanned",
            text=text.strip(),
        )

        return CanonicalDocument(
            document_id=doc_id,
            original_filename=file_path.name,
            metadata=metadata,
            total_pages=1,
            pages=[page],
            full_text=text.strip(),
        )
