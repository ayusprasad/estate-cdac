"""
DocuRAG — Semantic Chunker

Reads the extracted JSON from OpenDataLoader PDF, groups logical elements
into coherent chunks, and saves them to the DB.
"""
import json
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.database_models.document_model import Document
from src.database_models.page_model import Page
from src.database_models.chunk_model import Chunk
from src.database_models.shared_enums import ChunkType
from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)


async def chunk_document(document_id: str, db: AsyncSession) -> None:
    """
    Parse the extracted JSON for a document and create Chunk records.
    """
    logger.info("Starting chunking", document_id=str(document_id))
    
    # Fetch the document to get the extracted json path
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    
    if not document or not document.extra_metadata or "opendataloader_output_path" not in document.extra_metadata:
        logger.error("Document not extracted or missing JSON path", document_id=str(document_id))
        return
        
    output_dir = Path(document.extra_metadata["opendataloader_output_path"])
    file_path = Path(document.file_path)
    json_path = output_dir / f"{file_path.stem}.json"
    
    if not json_path.exists():
        logger.error("Extracted JSON not found", json_path=str(json_path))
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # First, fetch all Page records for this document so we can link chunks
    pages_result = await db.execute(select(Page).where(Page.document_id == document_id))
    pages = {p.page_number: p for p in pages_result.scalars().all()}
    
    new_chunks = []
    chunk_index = 0
    
    # Simple chunking: Each text block or table from opendataloader becomes a Chunk
    for el in data.get("kids", []):
        page_num = el.get("page number", 1)
        page_id = pages.get(page_num).id if page_num in pages else None
        
        text = el.get("content", "").strip()
        if not text:
            continue
            
        el_type = el.get("type", "text").lower()
        chunk_type = ChunkType.TEXT
        if "table" in el_type:
            chunk_type = ChunkType.TABLE
        elif "heading" in el_type:
            chunk_type = ChunkType.HEADING
        elif "list" in el_type:
            chunk_type = ChunkType.LIST
            
        chunk = Chunk(
            document_id=document.id,
            page_id=page_id,
            page_number=page_num,
            chunk_index=chunk_index,
            text=text,
            chunk_type=chunk_type,
            token_count=len(text.split())
        )
        new_chunks.append(chunk)
        chunk_index += 1
            
    db.add_all(new_chunks)
    await db.commit()
    logger.info("Chunking completed", document_id=str(document_id), chunks_created=len(new_chunks))
