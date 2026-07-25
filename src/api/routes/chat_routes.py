from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.database_models.database_connection import get_db
from src.retrieval.retriever import HybridRetriever
from src.llm.generator import RAGGenerator

router = APIRouter()

# Instantiate the generator once so the model can load in background
rag_generator = RAGGenerator()

class ChatQuery(BaseModel):
    query: str
    document_ids: Optional[list[str]] = None

class ChatResponse(BaseModel):
    answer: str

@router.post("/query", response_model=ChatResponse)
async def chat_query(request: ChatQuery, db=Depends(get_db)):
    try:
        retriever = HybridRetriever(db)
        
        # Phase 4: Hybrid Search
        top_chunks = await retriever.search(request.query, top_k=5)
        
        # Phase 5: RAG Generation
        answer = await rag_generator.generate_answer(request.query, top_chunks)
        
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
