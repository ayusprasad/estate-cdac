"""
DocuRAG — Canonical JSON Document Model

Defines the unified, standardized schema for extracted documents containing:
- Document Metadata (hash, checksum, filename, page count, file type)
- Pages (page number, page type, raw text, width, height)
- Sections (title, hierarchy level, body text, page range)
- Tables (headers, rows, caption, page number)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class CanonicalMetadata(BaseModel):
    """Document-level metadata."""
    title: Optional[str] = None
    source: Optional[str] = None
    language: Optional[str] = "en"
    checksum: Optional[str] = None
    file_type: str = "pdf"
    file_size_bytes: int = 0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: List[str] = Field(default_factory=list)


class CanonicalTable(BaseModel):
    """Extracted table representation."""
    table_id: str = Field(default_factory=lambda: str(uuid4()))
    page_number: int
    caption: Optional[str] = None
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    html_content: Optional[str] = None


class CanonicalSection(BaseModel):
    """Document section or heading block."""
    section_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    level: int = 1
    page_number: int
    text: str = ""


class CanonicalPage(BaseModel):
    """Single page representation."""
    page_number: int
    page_type: str = "digital"  # digital | scanned | mixed
    text: str = ""
    width: Optional[float] = None
    height: Optional[float] = None
    tables: List[CanonicalTable] = Field(default_factory=list)


class CanonicalDocument(BaseModel):
    """
    Canonical JSON Document Root Container.
    All extractors (PyMuPDF, pdfplumber, Tesseract/OCR, Pandas, JSON) convert
    input files into this schema.
    """
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    original_filename: str
    metadata: CanonicalMetadata = Field(default_factory=CanonicalMetadata)
    total_pages: int = 0
    pages: List[CanonicalPage] = Field(default_factory=list)
    sections: List[CanonicalSection] = Field(default_factory=list)
    tables: List[CanonicalTable] = Field(default_factory=list)
    full_text: str = ""

    def save_to_json(self, output_path: Path) -> Path:
        """Export the document to a canonical .json file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return output_path

    @classmethod
    def load_from_json(cls, json_path: Path) -> CanonicalDocument:
        """Load a CanonicalDocument from a .json file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)
