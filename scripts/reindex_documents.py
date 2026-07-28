"""
DocuRAG — Re-index All Uploaded Documents

Re-runs semantic chunking and vector embedding on all existing documents in the database.
Fixes isolated heading chunks and updates section_heading metadata.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from src.database_models.database_connection import AsyncSessionLocal
from src.database_models.document_model import Document
from src.database_models.page_model import Page
from src.database_models.chunk_model import Chunk
from src.database_models.processing_job_model import ProcessingJob
from src.document_processing.semantic_chunker import chunk_document
from src.document_processing.vector_embedder import embed_document_chunks
from application_configuration.logger_setup import get_logger

logger = get_logger("reindex_documents")


async def main():
    logger.info("Starting document re-indexing")
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Document))
        documents = result.scalars().all()
        
        logger.info(f"Found {len(documents)} documents to re-index")
        
        for doc in documents:
            logger.info(f"Re-indexing document: {doc.original_filename} (ID: {doc.id})")
            
            # Step 1: Re-chunk document with section-aware chunking
            await chunk_document(str(doc.id), db)
            
            # Step 2: Re-embed chunks with sentence transformer
            await embed_document_chunks(str(doc.id), db)
            
            logger.info(f"Re-indexing complete for: {doc.original_filename}")
            
    logger.info("All documents successfully re-indexed!")

if __name__ == "__main__":
    asyncio.run(main())
