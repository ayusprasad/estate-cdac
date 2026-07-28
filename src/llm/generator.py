"""
DocuRAG — Phase 8: Local RAG Generation Engine

Two-mode operation designed for i5-12500H + 16 GB RAM (CPU-only):

  Mode A — llama.cpp (GGUF model present):
    Uses llama-cpp-python to run quantized GGUF models locally.
    Recommended: Q4_K_M or Q5_K_M variants (~4-8 GB RAM).
    Thread count = LLM_N_THREADS (default 8, matching i5-12500H thread count).
    GPU layers = 0 (CPU-only, Intel Iris Xe shared memory not suitable for LLM).

  Mode B — Extractive fallback (no GGUF model):
    Zero-latency, zero-hallucination response constructed directly from the
    top retrieved chunks with full citations. No LLM required.
    This is what runs if ./models/llm/model.gguf does not exist.

The generator is stateless per-request. The llama.cpp Llama instance is
loaded once and cached as a module-level singleton to avoid reloading
the model weights on every request.

Prompt engineering principles:
  - Uses proper ChatML format (<|im_start|>/<|im_end|>) for Qwen2.5 compatibility
  - System prompt strictly instructs the model to answer ONLY from context
  - Context is injected as numbered citations [1], [2], etc.
  - Temperature = 0.1 (near-deterministic, reduces hallucination)
  - Max tokens = 1024 (sufficient for most answers, respects 16 GB RAM limit)
  - Chunk text is truncated at sentence boundaries to preserve coherence
"""
from __future__ import annotations

import asyncio
import re
import textwrap
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from application_configuration.environment_settings import get_settings
from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Maximum characters from each chunk inserted into the prompt.
# Keeps prompt within a safe context window for quantized models.
MAX_CHUNK_CHARS = 1200

# Approximate characters per token for budget estimation
CHARS_PER_TOKEN = 3.5

# Module-level llama.cpp instance (loaded once, reused on every request)
_llm = None
_llm_available = False


def _try_load_llama() -> None:
    """
    Attempt to load the llama.cpp model.
    Sets _llm and _llm_available at module level.
    Gracefully handles missing model file or missing llama-cpp-python package.
    """
    global _llm, _llm_available
    get_settings.cache_clear()
    current_settings = get_settings()

    model_path: Path = current_settings.llm.model_path.resolve()

    if not model_path.exists():
        logger.warning(
            "LLM model file not found — running in extractive fallback mode",
            model_path=str(model_path),
            hint="Download a GGUF model (Q4_K_M recommended) to ./models/llm/model.gguf",
        )
        _llm_available = False
        return

    # Skip loading if the filename indicates it's intentionally disabled
    if "disabled" in model_path.name.lower():
        logger.warning(
            "LLM model is disabled by name — running in extractive fallback mode",
            model_path=str(model_path),
            hint="Rename or replace the model file to enable LLM generation",
        )
        _llm_available = False
        return

    try:
        from llama_cpp import Llama  # type: ignore

        logger.info(
            "Loading llama.cpp GGUF model",
            model_path=str(model_path),
            n_threads=current_settings.llm.n_threads,
            n_ctx=current_settings.llm.context_size,
            n_gpu_layers=current_settings.llm.n_gpu_layers,
        )
        _llm = Llama(
            model_path=str(model_path),
            n_ctx=current_settings.llm.context_size,
            n_threads=current_settings.llm.n_threads,
            n_gpu_layers=current_settings.llm.n_gpu_layers,
            verbose=False,
        )
        _llm_available = True
        logger.info("llama.cpp model loaded successfully")

    except ImportError:
        logger.warning(
            "llama-cpp-python not installed — running in extractive fallback mode",
            hint="pip install llama-cpp-python to enable local LLM generation",
        )
        _llm_available = False

    except Exception as exc:
        logger.error(
            "Failed to load llama.cpp model",
            model_path=str(model_path),
            error=str(exc),
        )
        _llm_available = False


# Attempt model load at import time (happens once at server startup)
_try_load_llama()


# ── Text Utilities ─────────────────────────────────────────────────────────────

def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """
    Truncate text at the last complete sentence boundary within max_chars.
    Falls back to word boundary if no sentence boundary found.
    Avoids cutting mid-word or mid-sentence.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    # Try to find the last sentence boundary (. ! ?)
    last_sentence_end = -1
    for match in re.finditer(r'[.!?]\s', truncated):
        last_sentence_end = match.end()

    if last_sentence_end > max_chars * 0.5:  # At least half the text
        return truncated[:last_sentence_end].strip()

    # Fall back to last word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.3:
        return truncated[:last_space].strip() + "..."

    return truncated.strip() + "..."


# ── Prompt Builder (ChatML format for Qwen2.5) ────────────────────────────────

# System prompt with strict grounding rules
_SYSTEM_PROMPT = textwrap.dedent("""\
    You are DocuRAG, a document analysis assistant. Follow these rules strictly:

    1. ONLY answer using the numbered source excerpts provided below.
    2. If the sources do not contain enough information, respond EXACTLY with:
       "The uploaded documents do not contain sufficient information to answer this question."
    3. NEVER use your training knowledge, prior facts, or assumptions.
    4. After each claim or fact, cite the source number: [1], [2], etc.
    5. Be precise, factual, and concise.
    6. If multiple sources support a point, cite all relevant ones: [1][3].
    7. Do not repeat the question. Go straight to the answer.
""")


def _build_rag_prompt(query: str, chunks: List[Dict[str, Any]], enrichment_text: Optional[str] = None) -> str:
    """
    Build a ChatML formatted prompt for Qwen2.5 incorporating retrieved context,
    optional agent enrichment insights (e.g. mathematical proofs), and strict grounding instructions.
    """
    # Build numbered source context with metadata labels
    context_parts = []
    token_budget = int(settings.llm.context_size * CHARS_PER_TOKEN * 0.6)  # 60% for context
    used_chars = 0

    if enrichment_text:
        context_parts.append(enrichment_text)
        used_chars += len(enrichment_text)

    for chunk in chunks:
        rank = chunk.get("rank", "?")
        doc_name = chunk.get("document_name", "Unknown")
        page = chunk.get("page_number")
        section = chunk.get("section_title")
        text = chunk.get("text", "").strip()

        # Truncate at sentence boundary instead of hard character cut
        text = _truncate_at_sentence(text, MAX_CHUNK_CHARS)

        # Build location header
        loc_parts = [f"Document: {doc_name}"]
        if page:
            loc_parts.append(f"Page {page}")
        if section:
            loc_parts.append(f"§ {section}")
        location = " | ".join(loc_parts)

        chunk_block = f"[{rank}] {location}\n{text}"

        # Check token budget
        if used_chars + len(chunk_block) > token_budget:
            logger.debug("Token budget reached, skipping remaining chunks",
                         chunks_used=len(context_parts), chunks_total=len(chunks))
            break

        context_parts.append(chunk_block)
        used_chars += len(chunk_block)

    context_block = "\n\n".join(context_parts)

    # Build ChatML formatted prompt
    prompt = (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Sources:\n{context_block}\n\n"
        f"Question: {query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


# ── Extractive Fallback ────────────────────────────────────────────────────────

def _extractive_answer(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Build a zero-hallucination extractive answer directly from chunks.
    Used when no GGUF model is available.
    Filters out chunks that are completely irrelevant (score < -3.0).
    Formats output with proper citations for the frontend to render.
    """
    # Filter out highly irrelevant chunks (cross-encoder logits < -3.0)
    # Sigmoid-normalised scores (0-1) will always pass this safely.
    valid_chunks = [c for c in chunks if c.get("score", 0.0) >= -3.0]

    if not valid_chunks:
        return (
            "I could not find sufficient information in the uploaded documents "
            "to answer your question. Please ensure your query is related to "
            "the uploaded documents."
        )

    lines = [
        "Based on the uploaded documents, here are the most relevant excerpts:\n"
    ]
    for chunk in valid_chunks:
        rank = chunk.get("rank", "?")
        doc_name = chunk.get("document_name", "Unknown")
        page = chunk.get("page_number")
        section = chunk.get("section_title")
        text = chunk.get("text", "").strip()
        score = chunk.get("score", 0.0)

        citation_parts = [doc_name]
        if page:
            citation_parts.append(f"Page {page}")
        if section:
            citation_parts.append(f"§ {section}")
        citation = " — ".join(citation_parts)

        # Format relevance as percentage for readability
        if 0 <= score <= 1:
            relevance_str = f"{score * 100:.0f}%"
        else:
            relevance_str = f"{score:.3f}"

        lines.append(f"**[{rank}] {citation}** (relevance: {relevance_str})")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip()


# ── RAG Generator ──────────────────────────────────────────────────────────────

class RAGGenerator:
    """
    Phase 8: Local RAG Generation Engine.

    Generates grounded answers using:
      - llama.cpp GGUF model (if model file exists and llama-cpp-python installed)
      - Extractive fallback (if no model — zero-hallucination, zero-latency)

    Usage:
        generator = RAGGenerator()
        result = await generator.generate(query, chunks)
        print(result["answer"])     # The answer text
        print(result["mode"])       # "llm" or "extractive"
        print(result["citations"])  # list of citation dicts
    """

    def __init__(self) -> None:
        # Do NOT snapshot _llm_available here — read it live from the module
        # global so that monkeypatching in tests (and hot-reloading at runtime)
        # is reflected immediately without recreating the instance.
        pass

    @property
    def mode(self) -> str:
        """Return 'llm' if GGUF model is loaded, else 'extractive'."""
        return "llm" if _llm_available else "extractive"

    def _run_llm(self, prompt: str) -> str:
        """
        Call llama.cpp synchronously (blocking).
        Run via asyncio.to_thread() in async context to avoid blocking the event loop.
        """
        response = _llm(
            prompt,
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
            echo=False,
            stop=["<|im_end|>", "<|im_start|>"],
        )
        return response["choices"][0]["text"].strip()

    def _build_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citation metadata from chunks for the response."""
        citations = []
        for chunk in chunks:
            citations.append({
                "rank": chunk.get("rank"),
                "document_name": chunk.get("document_name"),
                "document_id": chunk.get("document_id"),
                "page_number": chunk.get("page_number"),
                "section_title": chunk.get("section_title"),
                "chunk_type": chunk.get("chunk_type"),
                "relevance_score": round(chunk.get("score", 0.0), 4),
            })
        return citations

    async def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate a grounded answer for the given query and retrieved chunks.

        Parameters
        ----------
        query  : The user's natural language question.
        chunks : List of CitedChunk dicts from SearchService.

        Returns
        -------
        dict with keys:
          answer      : str — the generated answer
          mode        : "llm" | "extractive"
          citations   : list of citation metadata dicts
          latency_ms  : float
          model       : str — model name or "extractive"
          llm_available : bool
        """
        t0 = time.perf_counter()

        if not chunks:
            answer = _extractive_answer(query, [])
            return {
                "answer": answer,
                "mode": "extractive",
                "citations": [],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "model": "extractive",
                "llm_available": False,
            }

        citations = self._build_citations(chunks)

        if _llm_available and _llm is not None:
            try:
                prompt = _build_rag_prompt(query, chunks)
                logger.info(
                    "Generating LLM answer",
                    query=query[:80],
                    chunks_count=len(chunks),
                    prompt_chars=len(prompt),
                )
                # Run blocking llama.cpp call in a thread pool
                answer = await asyncio.to_thread(self._run_llm, prompt)
                mode = "llm"
                model_name = str(settings.llm.model_path.name)

            except Exception as exc:
                logger.error(
                    "LLM generation failed — falling back to extractive",
                    error=str(exc),
                )
                answer = _extractive_answer(query, chunks)
                mode = "extractive"
                model_name = "extractive"
        else:
            answer = _extractive_answer(query, chunks)
            mode = "extractive"
            model_name = "extractive"

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "Answer generated",
            mode=mode,
            latency_ms=latency_ms,
            answer_chars=len(answer),
        )

        return {
            "answer": answer,
            "mode": mode,
            "citations": citations,
            "latency_ms": latency_ms,
            "model": model_name,
            "llm_available": _llm_available,
        }

    async def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Backward-compatible wrapper returning just the answer string.
        Used by existing chat_routes.py.
        """
        result = await self.generate(query, chunks)
        return result["answer"]


# ── Module-level singleton ────────────────────────────────────────────────────

_generator: Optional[RAGGenerator] = None


def get_generator() -> RAGGenerator:
    """Return cached RAGGenerator singleton."""
    global _generator
    if _generator is None:
        _generator = RAGGenerator()
    return _generator
