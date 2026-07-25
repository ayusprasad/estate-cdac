"""
DocuRAG — Document Extractor

Uses OpenDataLoader PDF to extract structured text, tables, and images from documents.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import opendataloader_pdf

from application_configuration.logger_setup import get_logger
from application_configuration.environment_settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


def run_opendataloader_extraction(file_path: Path, output_dir: Path) -> dict[str, Any]:
    """
    Extract text, tables, and layout from a PDF using opendataloader-pdf.
    
    Args:
        file_path: The path to the raw PDF file.
        output_dir: The directory where the extracted JSON and images should be saved.
        
    Returns:
        The parsed JSON structure containing pages, elements, tables, etc.
    """
    logger.info("Starting OpenDataLoader extraction", file_path=str(file_path))
    
    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert PDF to JSON with XY-Cut reading order and Cluster table detection
        opendataloader_pdf.convert(
            input_path=str(file_path),
            output_dir=str(output_dir),
            format="json",
            image_output="external",
            image_dir=str(image_dir),
            table_method="cluster", 
            reading_order="xycut",
            quiet=True,
        )
    except Exception as exc:
        logger.error("OpenDataLoader extraction failed", file_path=str(file_path), exc_info=exc)
        raise RuntimeError(f"Extraction failed: {exc}") from exc
    
    # OpenDataLoader writes the output file using the stem of the input file
    json_path = output_dir / f"{file_path.stem}.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Expected JSON output not found at {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    logger.info("OpenDataLoader extraction complete", file_path=str(file_path), pages=data.get("number of pages", 0))
    return data


async def process_document_extraction(document_id: str, session_maker) -> None:
    """
    Background task to execute Phase 2 extraction and save to the DB.
    """
    import asyncio
    from src.database_models.document_model import Document
    from src.database_models.page_model import Page
    from src.database_models.shared_enums import DocumentStatus
    from sqlalchemy import select
    
    logger.info("Starting background extraction", document_id=str(document_id))
    
    async with session_maker() as db:
        # Fetch the document
        result = await db.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none()
        
        if not document:
            logger.error("Document not found for extraction", document_id=str(document_id))
            return
            
        try:
            # Update status to extracting
            document.status = DocumentStatus.EXTRACTING
            await db.commit()
            
            # Run extraction in a separate thread because opendataloader blocks
            file_path = Path(document.file_path)
            output_dir = file_path.parent / f"{file_path.stem}_extracted"
            
            # Since convert is synchronous, we run it in a threadpool to avoid blocking FastAPI
            data = await asyncio.to_thread(
                run_opendataloader_extraction, 
                file_path, 
                output_dir
            )
            
            # Optional: Map the extracted JSON pages back to our Page ORM models
            # Here we just save the full extracted JSON to the document extra_metadata
            # for now, as full page mapping can be complex. 
            if not document.extra_metadata:
                document.extra_metadata = {}
            document.extra_metadata["opendataloader_output_path"] = str(output_dir)
            
            # Update status to extracted
            document.status = DocumentStatus.EXTRACTED
            await db.commit()
            logger.info("Extraction completed successfully", document_id=str(document_id))
            
            # Phase 3: Semantic Chunking
            from src.document_processing.semantic_chunker import chunk_document
            document.status = DocumentStatus.CHUNKING
            await db.commit()
            
            await chunk_document(document_id, db)
            
            document.status = DocumentStatus.CHUNKED
            await db.commit()
            
            # Phase 4: Vector Embedding
            from src.document_processing.vector_embedder import embed_document_chunks
            document.status = DocumentStatus.EMBEDDING
            await db.commit()
            
            await embed_document_chunks(document_id, db)
            
            document.status = DocumentStatus.READY
            await db.commit()
            logger.info("Full document ingestion pipeline completed!", document_id=str(document_id))
            
        except Exception as exc:
            logger.error("Background extraction failed", document_id=str(document_id), exc_info=exc)
            document.status = DocumentStatus.FAILED
            document.error_message = f"Extraction failed: {str(exc)}"
            await db.commit()

