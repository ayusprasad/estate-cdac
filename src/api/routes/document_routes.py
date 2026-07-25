"""
DocuRAG — Documents API Endpoints

Provides:
  POST /api/v1/documents/upload  — Upload and ingest a document
  GET  /api/v1/documents/        — List documents (paginated)
  GET  /api/v1/documents/{id}    — Get document details
  GET  /api/v1/documents/{id}/status — Get processing status
  GET  /api/v1/documents/{id}/pages  — Get page classification results
  DELETE /api/v1/documents/{id}  — Delete a document

All write operations are wrapped in DB transactions managed by get_db().
File uploads are streamed and never fully loaded into memory.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status, BackgroundTasks
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.document_processing.ingestion_pipeline import DocumentIngestionPipeline
from src.document_processing.processing_schemas import (
    DocumentClassificationResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadMetadata,
    DocumentUploadResponse,
    PageClassificationResult,
    ProcessingStatusResponse,
)
from src.database_models.database_connection import get_db
from src.database_models.document_model import Document
from src.database_models.page_model import Page
from src.database_models.shared_enums import DocumentStatus
from src.shared_utilities.custom_exceptions import (
    DuplicateDocumentError,
    FileSizeLimitError,
    IngestionError,
    UnsupportedFileTypeError,
    DocumentNotFoundError,
)
from application_configuration.logger_setup import get_logger
from application_configuration.environment_settings import get_settings

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for processing",
    description=(
        "Accepts a document file (PDF, DOCX, XLSX, CSV, or image) and enqueues it "
        "for the full processing pipeline. Returns immediately with a document_id. "
        "Poll GET /documents/{id}/status to track progress."
    ),
)
async def upload_document(
    file: Annotated[UploadFile, File(description="Document file to ingest")],
    title: Annotated[Optional[str], Form()] = None,
    tags: Annotated[Optional[str], Form(description="Comma-separated tags")] = None,
    source: Annotated[Optional[str], Form()] = None,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    """
    Upload and ingest a document.

    The file is:
    1. Streamed to a temporary location
    2. Validated (size + extension)
    3. Checksummed for deduplication
    4. Saved to raw storage
    5. Classified page-by-page
    6. Stored in the database

    All steps complete synchronously for Phase 1.
    Phase 2+ will offload extraction to Celery workers.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    logger.info(
        "Document upload received",
        filename=file.filename,
        content_type=file.content_type,
    )

    # Build extra metadata from form fields
    extra_metadata: dict = {}
    if title:
        extra_metadata["title"] = title
    if tags:
        extra_metadata["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if source:
        extra_metadata["source"] = source

    # Stream upload to a temp file (avoids loading large PDFs into RAM)
    with tempfile.NamedTemporaryFile(
        dir=settings.storage.temp_dir,
        delete=False,
        suffix=Path(file.filename).suffix,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        content = await file.read()
        tmp_file.write(content)

    try:
        pipeline = DocumentIngestionPipeline(db)
        response = await pipeline.ingest(
            file_path=tmp_path,
            original_filename=file.filename,
            extra_metadata=extra_metadata or None,
        )
        
        # Enqueue Phase 2 extraction in the background
        from fastapi import BackgroundTasks
        from src.document_processing.data_extractor import process_document_extraction
        
        # We need a new session context for the background task
        from src.database_models.database_connection import AsyncSessionLocal
        background_tasks.add_task(process_document_extraction, response.document_id, AsyncSessionLocal)
        
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except FileSizeLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except DuplicateDocumentError as exc:
        logger.info(f"Document already exists, returning existing ID {exc.existing_doc_id}")
        # Gracefully return the existing document to make the UX seamless
        result = await db.execute(select(Document).where(Document.id == exc.existing_doc_id))
        existing_doc = result.scalar_one()
        return DocumentUploadResponse(
            document_id=existing_doc.id,
            original_filename=existing_doc.original_filename,
            file_type=existing_doc.file_type,
            file_size_bytes=existing_doc.file_size_bytes,
            checksum=existing_doc.checksum,
            status=existing_doc.status,
        )
    except IngestionError as exc:
        logger.error("Ingestion pipeline failed", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    finally:
        # Always clean up the temporary file
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    return response


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all documents",
)
async def list_documents(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    status_filter: Annotated[Optional[DocumentStatus], Query(alias="status")] = None,
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """Return a paginated list of documents with optional status filter."""
    query = select(Document)
    count_query = select(func.count(Document.id))

    if status_filter:
        query = query.where(Document.status == status_filter)
        count_query = count_query.where(Document.status == status_filter)

    # Total count
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    # Paginated results
    offset = (page - 1) * page_size
    query = query.order_by(Document.created_at.desc()).offset(offset).limit(page_size)
    result = await db.execute(query)
    documents = result.scalars().all()

    return DocumentListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[DocumentResponse.model_validate(doc) for doc in documents],
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document details",
)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Return full details for a single document."""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )
    return DocumentResponse.model_validate(document)


@router.get(
    "/{document_id}/status",
    response_model=ProcessingStatusResponse,
    summary="Get processing status",
)
async def get_document_status(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ProcessingStatusResponse:
    """Return the current processing status of a document."""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )
    return ProcessingStatusResponse(
        document_id=document.id,
        status=document.status,
        error_message=document.error_message,
        processing_started_at=document.processing_started_at,
        processing_completed_at=document.processing_completed_at,
    )


@router.get(
    "/{document_id}/pages",
    response_model=DocumentClassificationResponse,
    summary="Get page classification results",
)
async def get_document_pages(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentClassificationResponse:
    """Return the per-page classification results for a processed document."""
    # Verify document exists
    doc_result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = doc_result.scalar_one_or_none()
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )

    # Fetch pages ordered by page number
    pages_result = await db.execute(
        select(Page)
        .where(Page.document_id == document_id)
        .order_by(Page.page_number)
    )
    pages = pages_result.scalars().all()

    page_results = [
        PageClassificationResult(
            page_number=p.page_number,
            page_type=p.page_type,
            text_char_count=0,  # Available post-extraction
            has_images=p.ocr_applied,
            width=p.width,
            height=p.height,
        )
        for p in pages
    ]

    from src.database_models.shared_enums import PageType
    return DocumentClassificationResponse(
        document_id=document_id,
        total_pages=len(page_results),
        digital_pages=sum(1 for p in page_results if p.page_type == PageType.DIGITAL),
        scanned_pages=sum(1 for p in page_results if p.page_type == PageType.SCANNED),
        mixed_pages=sum(1 for p in page_results if p.page_type == PageType.MIXED),
        page_results=page_results,
    )


from fastapi import Response

@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
    summary="Delete a document",
)
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a document and all associated data (cascaded)."""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )

    # Remove from raw storage
    try:
        raw_file = Path(document.file_path)
        if raw_file.exists():
            raw_file.unlink()
            logger.info("Raw file deleted", path=str(raw_file))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not delete raw file", path=document.file_path, exc_info=exc)

    await db.delete(document)
    logger.info("Document deleted", doc_id=str(document_id))
