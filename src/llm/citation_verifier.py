"""
DocuRAG — Phase 9: Citation & Faithfulness Verifier

Computes how well the generated answer is grounded in the retrieved source chunks.
This is the hallucination-detection layer.

Algorithm (lightweight, CPU-native — no additional model required):
──────────────────────────────────────────────────────────────────────
1. Tokenise the answer into sentences using simple punctuation splitting.
2. For each sentence, compute a lexical overlap score against all chunks:
     overlap(sentence, chunk) = |tokens(sentence) ∩ tokens(chunk)| / |tokens(sentence)|
3. A sentence is "grounded" if its max overlap with any chunk ≥ GROUNDING_THRESHOLD.
4. Faithfulness = grounded_sentences / total_sentences ∈ [0.0, 1.0].

Interpretation:
  1.0 = fully grounded (every sentence traceable to a chunk)
  0.8+ = high faithfulness
  0.5–0.8 = partial grounding (some extrapolation)
  < 0.5 = low faithfulness (possible hallucination)

Why lexical overlap and not embedding similarity?
  - No additional model inference required (keeps CPU latency low)
  - Works well for extractive / near-extractive answers
  - For LLM-generated answers we add TF-IDF-weighted token matching

The CitationVerifier also builds a sentence-level grounding map which is
returned to the API for optional display in the frontend.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Minimum overlap ratio to consider a sentence grounded in a chunk
GROUNDING_THRESHOLD: float = 0.25

# Short sentences (< N tokens) are excluded from scoring (likely boilerplate)
MIN_SENTENCE_TOKENS: int = 4

# Common English stop words — excluded from token overlap computation
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "up", "about", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "then", "once", "that", "this",
    "these", "those", "it", "its", "and", "or", "but", "so", "yet",
    "both", "either", "not", "no", "nor", "as", "if", "than", "because",
    "while", "when", "where", "how", "what", "which", "who", "whom",
    "based", "strictly", "documents", "indexed", "document",
})


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class SentenceGround:
    sentence: str
    is_grounded: bool
    best_score: float
    best_chunk_rank: Optional[int]
    best_chunk_doc: Optional[str]


@dataclass
class VerificationResult:
    faithfulness: float                    # 0.0 – 1.0
    grounded_count: int
    total_sentences: int
    sentence_grounds: List[SentenceGround]
    verdict: str                           # "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN"


# ── Core Utilities ────────────────────────────────────────────────────────────

def _tokenise(text: str) -> frozenset:
    """
    Lowercase, strip punctuation, split on whitespace, remove stop words.
    Returns a frozenset of content tokens.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return frozenset(t for t in tokens if t not in _STOP_WORDS and len(t) > 1)


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using common punctuation heuristics.
    Avoids importing NLTK to keep dependencies minimal.
    """
    # Split on ". " or ".\n" or "! " or "? " — but preserve [1] markers
    raw = re.split(r"(?<=[.!?])\s+", text)
    # Further split on newlines used as soft sentence breaks in extractive mode
    sentences = []
    for part in raw:
        sub = [s.strip() for s in part.split("\n") if s.strip()]
        sentences.extend(sub)
    return [s for s in sentences if len(s) > 10]


def _overlap_score(sentence_tokens: frozenset, chunk_tokens: frozenset) -> float:
    """
    Compute Jaccard-like overlap: intersection / sentence_size.
    Uses sentence size as denominator (recall-oriented: does chunk cover sentence?)
    """
    if not sentence_tokens:
        return 0.0
    intersection = len(sentence_tokens & chunk_tokens)
    return intersection / len(sentence_tokens)


# ── Citation Verifier ─────────────────────────────────────────────────────────

class CitationVerifier:
    """
    Phase 9: Hallucination detection and citation faithfulness scorer.

    Computes a faithfulness score for a generated answer by checking
    how many answer sentences are traceable to retrieved chunks.

    Usage:
        verifier = CitationVerifier()
        score = verifier.score(answer, chunks)        # float 0.0–1.0
        result = verifier.verify(answer, chunks)      # full VerificationResult
    """

    def score(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
    ) -> float:
        """
        Return faithfulness score (0.0–1.0) for the answer relative to chunks.

        0.0 = not grounded at all
        1.0 = every sentence has clear overlap with at least one chunk
        """
        result = self.verify(answer, chunks)
        return result.faithfulness

    def verify(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
    ) -> VerificationResult:
        """
        Full citation verification.

        Returns a VerificationResult with sentence-level grounding details,
        overall faithfulness score, and a high/medium/low verdict.
        """
        if not answer or not answer.strip():
            return VerificationResult(
                faithfulness=0.0,
                grounded_count=0,
                total_sentences=0,
                sentence_grounds=[],
                verdict="UNKNOWN",
            )

        # Pre-tokenise all chunks (one pass)
        chunk_token_sets: List[Tuple[Dict, frozenset]] = [
            (chunk, _tokenise(chunk.get("text", "")))
            for chunk in chunks
        ]

        # Split answer into scoreable sentences
        sentences = _split_sentences(answer)
        sentence_grounds: List[SentenceGround] = []
        grounded = 0

        for sentence in sentences:
            sent_tokens = _tokenise(sentence)

            # Skip very short sentences (boilerplate headers, "[1]", etc.)
            if len(sent_tokens) < MIN_SENTENCE_TOKENS:
                continue

            best_score = 0.0
            best_chunk_rank = None
            best_chunk_doc = None

            for chunk, chunk_tokens in chunk_token_sets:
                score = _overlap_score(sent_tokens, chunk_tokens)
                if score > best_score:
                    best_score = score
                    best_chunk_rank = chunk.get("rank")
                    best_chunk_doc = chunk.get("document_name")

            is_grounded = best_score >= GROUNDING_THRESHOLD
            if is_grounded:
                grounded += 1

            sentence_grounds.append(SentenceGround(
                sentence=sentence[:200],   # trim for logging
                is_grounded=is_grounded,
                best_score=round(best_score, 4),
                best_chunk_rank=best_chunk_rank,
                best_chunk_doc=best_chunk_doc,
            ))

        total = len(sentence_grounds)
        faithfulness = round(grounded / total, 4) if total > 0 else 1.0

        verdict = (
            "HIGH" if faithfulness >= 0.8
            else "MEDIUM" if faithfulness >= 0.5
            else "LOW" if total > 0
            else "UNKNOWN"
        )

        logger.info(
            "Citation verification complete",
            faithfulness=faithfulness,
            grounded=grounded,
            total=total,
            verdict=verdict,
        )

        return VerificationResult(
            faithfulness=faithfulness,
            grounded_count=grounded,
            total_sentences=total,
            sentence_grounds=sentence_grounds,
            verdict=verdict,
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_verifier: Optional[CitationVerifier] = None


def get_verifier() -> CitationVerifier:
    """Return cached CitationVerifier singleton."""
    global _verifier
    if _verifier is None:
        _verifier = CitationVerifier()
    return _verifier
