"""
DocuRAG — Embedder Pipeline

Computes semantic vectors for extracted chunks using CPU-optimised
Sentence Transformers (llama.cpp could also be used here if GGUF models are preferred).
"""
import datetime
import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sentence_transformers import SentenceTransformer

# Enforce strict offline mode to prevent HuggingFace connection retries
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.database_models.chunk_model import Chunk
from application_configuration.logger_setup import get_logger
from application_configuration.environment_settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

# Global cache for the embedding model to avoid reloading it on every call
_model = None
MODEL_NAME = settings.embedding.model

def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading embedding model {MODEL_NAME} (Offline Mode)...")
        try:
            _model = SentenceTransformer(MODEL_NAME, device='cpu', local_files_only=True)
        except Exception as err:
            logger.warning("local_files_only load failed, falling back to default", error=str(err))
            _model = SentenceTransformer(MODEL_NAME, device='cpu')
    return _model


async def embed_document_chunks(document_id: str, db: AsyncSession) -> None:
    """
    Fetch all chunks for a document, compute embeddings, and save back to DB.
    """
    logger.info("Starting embedding generation", document_id=str(document_id))
    
    import asyncio
    model = await asyncio.to_thread(get_model)
    
    # Fetch all chunks for this document that don't have embeddings yet
    from uuid import UUID
    doc_uuid = UUID(document_id) if isinstance(document_id, str) else document_id
    result = await db.execute(
        select(Chunk)
        .where(Chunk.document_id == doc_uuid)
        .where(Chunk.embedding.is_(None))
    )
    chunks = result.scalars().all()
    
    if not chunks:
        logger.info("No chunks require embedding", document_id=str(document_id))
        return
        
    # Extract text from chunks
    texts = [chunk.text for chunk in chunks]
    
    # Compute embeddings in a thread to not block asyncio
    logger.info(f"Computing embeddings for {len(chunks)} chunks...")
    embeddings = await asyncio.to_thread(model.encode, texts, show_progress_bar=False)
    
    # Map back to chunks
    for i, chunk in enumerate(chunks):
        chunk.embedding = embeddings[i].tolist()
        chunk.embedding_model = MODEL_NAME
        # Use timezone-aware UTC datetime
        chunk.embedding_generated_at = datetime.datetime.now(datetime.timezone.utc)
        
    await db.commit()
    logger.info("Embedding generation completed", document_id=str(document_id), chunks_embedded=len(chunks))
