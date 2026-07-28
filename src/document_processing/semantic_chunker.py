"""
DocuRAG — Semantic Chunker

Reads the extracted JSON from OpenDataLoader PDF, groups logical elements
into coherent chunks (with section headings attached), and saves them to the DB.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.database_models.document_model import Document
from src.database_models.page_model import Page
from src.database_models.chunk_model import Chunk
from src.database_models.shared_enums import ChunkType
from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)

# Target chunk size in tokens (words) for grouping elements into coherent retrieval units
TARGET_CHUNK_TOKENS = 350
MIN_CHUNK_TOKENS = 30


async def chunk_document(document_id: str, db: AsyncSession) -> None:
    """
    Parse the extracted JSON for a document and create Chunk records.
    Groups headings and body content into context-rich retrieval units.
    Attaches section headings to all chunks within a section.
    """
    logger.info("Starting semantic chunking", document_id=str(document_id))
    
    # Fetch the document
    from uuid import UUID
    doc_uuid = UUID(document_id) if isinstance(document_id, str) else document_id
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
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
        
    pages_result = await db.execute(select(Page).where(Page.document_id == doc_uuid))
    pages = {p.page_number: p for p in pages_result.scalars().all()}
    
    # Delete any existing chunks for this document if re-chunking
    await db.execute(delete(Chunk).where(Chunk.document_id == doc_uuid))
    await db.commit()

    new_chunks: List[Chunk] = []
    chunk_index = 0
    current_section_heading: Optional[str] = None
    
    # Recursively extract raw elements from OpenDataLoader kids/list items
    raw_elements: List[Dict[str, Any]] = []
    
    def _extract_elements(element_list: list):
        for el in element_list:
            text = el.get("content", "").strip()
            el_type = str(el.get("type", "text")).lower()
            page_num = el.get("page number", 1)
            
            if text:
                raw_elements.append({
                    "type": el_type,
                    "content": text,
                    "page_number": page_num,
                })
            
            # Recurse into nested items
            if "list" in el_type and "list items" in el:
                _extract_elements(el["list items"])
            elif "kids" in el and isinstance(el["kids"], list):
                _extract_elements(el["kids"])

    _extract_elements(data.get("kids", []))
    
    # Group elements into coherent chunks
    current_chunk_texts: List[str] = []
    current_chunk_tokens = 0
    current_chunk_page = 1
    current_chunk_type = ChunkType.TEXT
    
    def _flush_current_chunk():
        nonlocal chunk_index, current_chunk_texts, current_chunk_tokens, current_chunk_page, current_chunk_type
        if not current_chunk_texts:
            return
            
        combined_text = "\n\n".join(current_chunk_texts).strip()
        if not combined_text:
            return
            
        page_obj = pages.get(current_chunk_page)
        page_id = page_obj.id if page_obj else None
        
        chunk = Chunk(
            document_id=document.id,
            page_id=page_id,
            page_number=current_chunk_page,
            section_title=current_section_heading,
            chunk_index=chunk_index,
            text=combined_text,
            chunk_type=current_chunk_type,
            token_count=len(combined_text.split()),
        )
        new_chunks.append(chunk)
        chunk_index += 1
        
        current_chunk_texts = []
        current_chunk_tokens = 0
        current_chunk_type = ChunkType.TEXT

    for el in raw_elements:
        el_type = el["type"]
        text = el["content"]
        page_num = el["page_number"]
        
        # Heading element: update section heading and start section
        if "heading" in el_type or "title" in el_type:
            current_section_heading = text
            formatted_text = f"## {text}"
            
            # Flush previous chunk if it has sufficient content
            if current_chunk_tokens >= MIN_CHUNK_TOKENS:
                _flush_current_chunk()
                
            current_chunk_texts.append(formatted_text)
            current_chunk_tokens += len(formatted_text.split())
            current_chunk_page = page_num
            current_chunk_type = ChunkType.HEADING
            continue
            
        # Determine element type
        el_chunk_type = ChunkType.TEXT
        if "table" in el_type:
            el_chunk_type = ChunkType.TABLE
        elif "list" in el_type:
            el_chunk_type = ChunkType.LIST
            
        tokens = len(text.split())
        
        # Flush if target tokens exceeded or page changed significantly
        if (current_chunk_tokens + tokens > TARGET_CHUNK_TOKENS and current_chunk_tokens >= MIN_CHUNK_TOKENS) or (page_num != current_chunk_page and current_chunk_tokens >= TARGET_CHUNK_TOKENS // 2):
            _flush_current_chunk()
            
        current_chunk_texts.append(text)
        current_chunk_tokens += tokens
        current_chunk_page = page_num
        if el_chunk_type != ChunkType.TEXT:
            current_chunk_type = el_chunk_type

    # Flush remaining buffer
    _flush_current_chunk()
    
    db.add_all(new_chunks)
    await db.commit()
    logger.info("Semantic chunking completed", document_id=str(document_id), chunks_created=len(new_chunks))
