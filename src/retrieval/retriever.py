import math
import numpy as np
from typing import List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.database_models.chunk_model import Chunk
from src.document_processing.vector_embedder import get_model
from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Phase 4: Hybrid Search Engine.
    Combines Vector Similarity (Dense) and BM25 (Sparse) for optimal retrieval.
    """
    def __init__(self, db: AsyncSession):
        self.db = db
        self.model = get_model()

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _compute_bm25(self, query: str, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75) -> List[float]:
        """Simple BM25 implementation."""
        query_terms = query.lower().split()
        if not query_terms:
            return [0.0] * len(chunks)

        # Tokenize chunks
        tokenized_chunks = [c.text.lower().split() for c in chunks]
        
        # Calculate DF and IDF
        df = {}
        for tokens in tokenized_chunks:
            unique_tokens = set(tokens)
            for qt in query_terms:
                if qt in unique_tokens:
                    df[qt] = df.get(qt, 0) + 1
                    
        N = len(chunks)
        idf = {qt: math.log(1 + (N - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5)) for qt in query_terms}
        
        # Calculate average document length
        avgdl = sum(len(t) for t in tokenized_chunks) / N if N > 0 else 0
        
        scores = []
        for tokens in tokenized_chunks:
            score = 0.0
            doc_len = len(tokens)
            tf = {qt: tokens.count(qt) for qt in query_terms}
            for qt in query_terms:
                if qt in tf and tf[qt] > 0:
                    term_tf = tf[qt]
                    score += idf[qt] * (term_tf * (k1 + 1)) / (term_tf + k1 * (1 - b + b * doc_len / (avgdl or 1)))
            scores.append(score)
            
        return scores

    def _normalize(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_val, max_val = min(scores), max(scores)
        if max_val == min_val:
            return [1.0 if max_val > 0 else 0.0 for _ in scores]
        return [(s - min_val) / (max_val - min_val) for s in scores]

    async def search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> List[Chunk]:
        """
        Executes hybrid search and returns top chunks.
        alpha: Weight for vector search (0.0 to 1.0). 1-alpha is BM25 weight.
        """
        import asyncio
        logger.info("Executing Hybrid Search", query=query)
        
        # 1. Fetch chunks
        stmt = select(Chunk).where(Chunk.embedding.is_not(None))
        result = await self.db.execute(stmt)
        chunks = result.scalars().all()
        
        if not chunks:
            logger.warning("No chunks found in database.")
            return []
            
        # 2. Vector Search (Dense) using highly-optimized numpy vectorization
        query_embedding = await asyncio.to_thread(self.model.encode, query)
        query_embedding = np.array(query_embedding, dtype=np.float64)
        
        # Build matrix of all chunk embeddings
        chunk_embeddings = np.array([c.embedding for c in chunks], dtype=np.float64)
        
        # Vectorized Cosine Similarity
        dots = np.dot(chunk_embeddings, query_embedding)
        query_norm = np.linalg.norm(query_embedding)
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
        
        denom = query_norm * chunk_norms
        denom[denom == 0] = 1e-9
        
        dense_scores = (dots / denom).tolist()
            
        # 3. Keyword Search (BM25 Sparse)
        sparse_scores = self._compute_bm25(query, chunks)
        
        # 4. Normalize and Combine
        norm_dense = self._normalize(dense_scores)
        norm_sparse = self._normalize(sparse_scores)
        
        combined_scores = []
        for i in range(len(chunks)):
            final_score = (alpha * norm_dense[i]) + ((1 - alpha) * norm_sparse[i])
            combined_scores.append((final_score, chunks[i]))
            
        # 5. Sort and return fast
        combined_scores.sort(key=lambda x: x[0], reverse=True)
        return [c for score, c in combined_scores[:top_k] if score > 0.05]
