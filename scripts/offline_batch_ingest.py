"""
DocuRAG — Offline Batch Ingestion CLI Tool

Batch ingests large folders of documents (PDFs, scanned PDFs, Excel, CSV, JSON, images)
offline according to the architecture flowchart:
1. Multi-Format Unified Parser -> Canonical JSON Document Model
2. Deduplication check via SHA256 file checksum
3. Semantic Chunker (Groups headings with section text into ~350-word chunks)
4. Vector Embedder (BGE model with GPU acceleration)
5. SQLite/Database Storage with READY status

Usage:
  .venv\\Scripts\\python.exe scripts/offline_batch_ingest.py --dir "C:\\path\\to\\documents"
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from uuid import UUID, uuid4

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sys
sys.stdout.reconfigure(encoding='utf-8')

from sqlalchemy import select
from src.database_models.database_connection import AsyncSessionLocal
from src.database_models.document_model import Document
from src.database_models.processing_job_model import ProcessingJob  # noqa: F401
from src.database_models.chunk_model import Chunk  # noqa: F401
from src.database_models.page_model import Page  # noqa: F401
from src.database_models.shared_enums import DocumentStatus
from src.document_processing.canonical_json_model import CanonicalDocument
from src.document_processing.semantic_chunker import chunk_document
from src.document_processing.unified_parser import UnifiedDocumentParser, compute_file_sha256
from src.document_processing.vector_embedder import embed_document_chunks
from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls", ".json", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


async def ingest_single_file(file_path: Path, parser: UnifiedDocumentParser) -> bool:
    """Ingest a single document file end-to-end."""
    file_path = file_path.resolve()
    logger.info("Processing file", path=str(file_path))

    checksum = compute_file_sha256(file_path)

    async with AsyncSessionLocal() as db:
        # Check deduplication
        existing = await db.execute(select(Document).where(Document.checksum == checksum))
        if existing.scalar_one_or_none():
            logger.info("Skipping duplicate document", filename=file_path.name, checksum=checksum)
            return False

        doc_id = str(uuid4())
        doc_uuid = UUID(doc_id)

        # 1. Parse to Canonical JSON
        canonical_doc: CanonicalDocument = parser.parse(file_path, document_id=doc_id)

        # Save canonical JSON artifact
        json_output_dir = PROJECT_ROOT / "data" / "processed" / "canonical_json"
        canonical_json_path = json_output_dir / f"{doc_id}.canonical.json"
        canonical_doc.save_to_json(canonical_json_path)

        # 2. Save raw file to raw storage
        raw_storage_dir = PROJECT_ROOT / "data" / "raw"
        raw_storage_dir.mkdir(parents=True, exist_ok=True)
        raw_file_dest = raw_storage_dir / f"{doc_id}{file_path.suffix}"
        
        with open(file_path, "rb") as src_f, open(raw_file_dest, "wb") as dst_f:
            dst_f.write(src_f.read())

        from src.database_models.shared_enums import DocumentType
        ext_clean = file_path.suffix.lstrip(".").lower()
        if ext_clean == "pdf":
            file_type_val = DocumentType.PDF
        elif ext_clean == "docx":
            file_type_val = DocumentType.DOCX
        elif ext_clean == "xlsx":
            file_type_val = DocumentType.XLSX
        elif ext_clean == "csv":
            file_type_val = DocumentType.CSV
        elif ext_clean in ("png", "jpg", "jpeg", "tiff", "bmp"):
            file_type_val = DocumentType.IMAGE
        else:
            file_type_val = DocumentType.UNKNOWN

        import mimetypes
        mime_val = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

        # 3. Create Document ORM record
        doc_record = Document(
            id=doc_uuid,
            original_filename=file_path.name,
            stored_filename=raw_file_dest.name,
            file_path=str(raw_file_dest),
            file_type=file_type_val,
            mime_type=mime_val,
            file_size_bytes=file_path.stat().st_size,
            checksum=checksum,
            page_count=canonical_doc.total_pages,
            status=DocumentStatus.EXTRACTED,
            extra_metadata={
                "canonical_json": str(canonical_json_path),
                "title": canonical_doc.metadata.title,
            },
        )
        db.add(doc_record)
        await db.commit()

        # 4. Semantic Chunker
        doc_record.status = DocumentStatus.CHUNKING
        await db.commit()

        # We pass canonical_doc text or full text to semantic chunker
        await chunk_document(str(doc_uuid), db)

        doc_record.status = DocumentStatus.CHUNKED
        await db.commit()

        # 5. Vector Embedder
        doc_record.status = DocumentStatus.EMBEDDING
        await db.commit()

        await embed_document_chunks(str(doc_uuid), db)

        # Final Status: READY
        doc_record.status = DocumentStatus.READY
        await db.commit()

        logger.info("Successfully ingested document!", filename=file_path.name, doc_id=doc_id)
        return True


async def batch_ingest_directory(dir_path: Path):
    """Scan and batch ingest all supported files in dir_path."""
    dir_path = Path(dir_path).resolve()
    if not dir_path.exists():
        print(f"Error: Directory does not exist: {dir_path}")
        sys.exit(1)

    print(f"\n========================================================")
    print(f"  DocuRAG — Offline Batch Document Ingestion Tool")
    print(f"  Scanning Directory: {dir_path}")
    print(f"========================================================\n")

    files_to_ingest: list[Path] = []
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files_to_ingest.append(p)

    if not files_to_ingest:
        print("No supported documents found to ingest.")
        return

    print(f"Found {len(files_to_ingest)} documents to process.")
    start_time = time.time()
    parser = UnifiedDocumentParser(use_ocr=True)

    success_count = 0
    duplicate_count = 0
    fail_count = 0

    for idx, fpath in enumerate(files_to_ingest, start=1):
        print(f"[{idx}/{len(files_to_ingest)}] Processing: {fpath.name}...")
        try:
            res = await ingest_single_file(fpath, parser)
            if res:
                success_count += 1
            else:
                duplicate_count += 1
        except Exception as exc:
            logger.error("Failed to ingest file", path=str(fpath), exc_info=exc)
            print(f"  ❌ FAILED: {exc}")
            fail_count += 1

    elapsed = time.time() - start_time
    print(f"\n========================================================")
    print(f"  Ingestion Complete in {elapsed:.2f} seconds!")
    print(f"  - Successfully Ingested & Embedded: {success_count}")
    print(f"  - Skipped (Duplicates): {duplicate_count}")
    print(f"  - Failed: {fail_count}")
    print(f"========================================================\n")


def main():
    parser = argparse.ArgumentParser(description="DocuRAG Offline Batch Ingestion Tool")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to folder containing documents (PDFs, Excel, CSV, JSON, images)",
    )
    args = parser.parse_args()
    asyncio.run(batch_ingest_directory(Path(args.dir)))


if __name__ == "__main__":
    main()
