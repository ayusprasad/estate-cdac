"""
DocuRAG — Chat API Routes (Phase 8 integrated)

POST /api/v1/chat/query — Full RAG pipeline:
    Phase 6 (Query Planner) →
    Phase 5/7 (Hybrid Search or SQL Agent) →
    Phase 8 (RAG Generator — llama.cpp or extractive fallback) →
    Phase 9 (Citation Verifier — faithfulness score)

The response now includes:
  - answer      : LLM-generated or extractive answer
  - mode        : "llm" | "extractive"
  - citations   : structured list with doc/page/section references
  - faithfulness: Phase 9 grounding score (0.0–1.0)
  - sources     : raw chunk list for downstream use
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.database_models.database_connection import get_db
from src.retrieval.search_router import SearchRouter
from src.llm.generator import get_generator
from src.llm.citation_verifier import CitationVerifier
from application_configuration.logger_setup import get_logger

router = APIRouter()
logger = get_logger(__name__)
_verifier = CitationVerifier()


# ── Request / Response Models ─────────────────────────────────────────────────

class ChatQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048,
                       description="Natural language question")
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Restrict search to specific document IDs"
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=20,
        description="Max number of chunks to retrieve"
    )
    db_label: Optional[str] = Field(
        default=None,
        description="Target SQL database label (Phase 7). Omit for document search."
    )


class CitationItem(BaseModel):
    rank: Optional[int]
    document_name: Optional[str]
    document_id: Optional[str]
    page_number: Optional[int]
    section_title: Optional[str]
    chunk_type: Optional[str]
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    mode: str                          # "llm" | "extractive"
    intent: str
    faithfulness: float                # Phase 9 grounding score
    citations: List[CitationItem]
    latency_ms: float
    generation_ms: float
    sources: List[dict]                # raw chunks (backward-compatible)
    llm_available: bool


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Full RAG pipeline: retrieve + generate + verify",
    description=(
        "Runs the complete pipeline: intent detection → hybrid search or SQL → "
        "local LLM generation (or extractive fallback) → citation grounding check. "
        "Terminal shows retrieval, generation, and faithfulness logs in real time."
    ),
    tags=["chat"],
)
async def chat_query(request: ChatQuery, db=Depends(get_db)):
    """
    Full RAG endpoint.

    Terminal logs show:
      - Intent detected (Phase 6)
      - Hybrid search timing (Phase 5)
      - Generation mode: llm / extractive (Phase 8)
      - Faithfulness score (Phase 9)
    """
    try:
        # ── Phase 6 + 5/7: Retrieve ───────────────────────────────────────
        router_instance = SearchRouter(db)
        result = await router_instance.route(
            query=request.query,
            document_ids=request.document_ids,
            top_k_override=request.top_k,
            db_label=request.db_label,
        )

        # ── Phase 8: Generate ─────────────────────────────────────────────
        generator = get_generator()
        gen_result = await generator.generate(
            query=request.query,
            chunks=result.chunks,
        )

        # ── Phase 9: Verify citation grounding ────────────────────────────
        faithfulness = _verifier.score(
            answer=gen_result["answer"],
            chunks=result.chunks,
        )

        logger.info(
            "RAG pipeline complete",
            intent=result.intent,
            mode=gen_result["mode"],
            chunks_used=len(result.chunks),
            faithfulness=faithfulness,
            generation_ms=gen_result["latency_ms"],
        )

        # Build structured citation list
        citations = [CitationItem(**c) for c in gen_result.get("citations", [])]

        return ChatResponse(
            answer=gen_result["answer"],
            mode=gen_result["mode"],
            intent=result.intent,
            faithfulness=faithfulness,
            citations=citations,
            latency_ms=result.latency_ms + gen_result["latency_ms"],
            generation_ms=gen_result["latency_ms"],
            sources=result.chunks,
            llm_available=gen_result["llm_available"],
        )

    except Exception as exc:
        logger.error("Chat query failed", error=str(exc), exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/status",
    summary="LLM engine status",
    tags=["chat"],
)
async def llm_status():
    """
    Return current LLM mode and model information.
    Useful for the frontend to show whether LLM generation is active.
    """
    generator = get_generator()
    model_path = str(__import__("application_configuration.environment_settings",
                                fromlist=["get_settings"]).get_settings().llm.model_path)
    return {
        "mode": generator.mode,
        "llm_available": generator.mode == "llm",
        "model_path": model_path,
        "hint": (
            "Place a GGUF model at ./models/llm/model.gguf and install "
            "llama-cpp-python to enable local LLM generation."
            if generator.mode == "extractive"
            else "LLM active."
        ),
    }
