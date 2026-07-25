import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
import asyncio

from src.database_models.chunk_model import Chunk
from src.document_processing.vector_embedder import get_model
from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

async def hybrid_search(query: str, db: AsyncSession, document_ids: Optional[list[str]] = None, top_k: int = 5) -> str:
    logger.info("Running hybrid search", query=query)
    
    # Get model and compute query embedding
    model = await asyncio.to_thread(get_model)
    query_embedding = await asyncio.to_thread(model.encode, query)
    query_embedding = query_embedding.astype(np.float64) 
    
    # Fetch chunks
    stmt = select(Chunk).where(Chunk.embedding.is_not(None))
        
    result = await db.execute(stmt)
    chunks = result.scalars().all()
    
    if not chunks:
        return "I could not find any relevant information in the uploaded documents."
        
    scored_chunks = []
    
    for chunk in chunks:
        # Hybrid scoring: simple BM25 mock (keyword match count) + cosine sim
        text_lower = chunk.text.lower()
        query_terms = query.lower().split()
        keyword_score = sum(1 for term in query_terms if term in text_lower) / len(query_terms) if query_terms else 0
        
        chunk_embedding = np.array(chunk.embedding, dtype=np.float64)
        vec_score = cosine_similarity(query_embedding, chunk_embedding)
        
        # Combine scores (weighting vector more heavily)
        final_score = (vec_score * 0.8) + (keyword_score * 0.2)
        scored_chunks.append((final_score, chunk))
        
    # Sort descending
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for score, c in scored_chunks[:top_k]]
    
    if not top_chunks or scored_chunks[0][0] < 0.2: # arbitrary low threshold
        return "I could not find any relevant information in the uploaded documents."
        
    # Construct answer ensuring NO hallucination
    answer = "Based strictly on the extracted document knowledge, here is the relevant information:\n\n"
    for i, chunk in enumerate(top_chunks):
        page_ref = f"(Page {chunk.page_number})" if chunk.page_number else ""
        answer += f"**Citation {i+1}** {page_ref}: {chunk.text.strip()}\n\n"
        
    return answer
