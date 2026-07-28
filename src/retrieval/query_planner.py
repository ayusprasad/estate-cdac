"""
DocuRAG — Phase 6: Query Planner & Intent Router

This module analyses every incoming query and produces a QueryPlan that tells
the downstream execution layer how to answer it.

Design principle: All routing is done with lightweight, zero-latency heuristics
(regex, keyword matching, syntactic patterns). No LLM is involved at this stage.
The goal is to select the right retrieval strategy in <5 ms before any DB
or model call is made.

Intent taxonomy
───────────────
┌──────────────────┬────────────────────────────────────────────────────────┐
│ Intent           │ Description                                            │
├──────────────────┼────────────────────────────────────────────────────────┤
│ FACTUAL          │ Definition / explanation questions ("what is X")       │
│ ANALYTICAL       │ Comparison, analysis, pros/cons ("compare X and Y")   │
│ NUMERICAL        │ Calculation, formula, statistics ("how much", %)      │
│ TABULAR          │ Data from tables or spreadsheets                       │
│ SQL_DATA         │ Structured database query (future SQL agent hook)      │
│ MULTILINGUAL     │ Query contains non-ASCII / non-English text            │
│ SUMMARISATION    │ Broad summary of a document or section                 │
│ IMAGE            │ Question about a figure, diagram, chart                │
│ GENERAL          │ Catch-all; uses standard hybrid retrieval              │
└──────────────────┴────────────────────────────────────────────────────────┘

Retrieval strategy per intent
──────────────────────────────
FACTUAL      → standard hybrid retrieval, top_k=5, alpha=0.7
ANALYTICAL   → wider search, top_k=8, alpha=0.6 (more BM25 keyword weight)
NUMERICAL    → chunk_types=["table", "formula"], top_k=5, alpha=0.5
TABULAR      → chunk_types=["table"], top_k=8, alpha=0.4
SQL_DATA     → flagged for SQL agent (no vector search)
MULTILINGUAL → alpha=0.8 (higher vector weight for cross-lingual matching)
SUMMARISATION→ top_k=15, alpha=0.65 (more context, higher recall)
IMAGE        → chunk_types=["caption"], top_k=5
GENERAL      → default hybrid
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)


# ── Intent Types ──────────────────────────────────────────────────────────────

class QueryIntent(str, Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    NUMERICAL = "numerical"
    TABULAR = "tabular"
    SQL_DATA = "sql_data"
    MULTILINGUAL = "multilingual"
    SUMMARISATION = "summarisation"
    IMAGE = "image"
    GENERAL = "general"


# ── Retrieval Strategy ────────────────────────────────────────────────────────

@dataclass
class RetrievalStrategy:
    """
    Parameters passed to SearchService.search() based on detected intent.
    """
    top_k: int = 5
    alpha: float = 0.7               # 0=BM25 only, 1=dense only
    use_reranker: bool = True
    chunk_types: Optional[List[str]] = None   # None = all chunk types
    use_sql_agent: bool = False


# ── Query Plan ────────────────────────────────────────────────────────────────

@dataclass
class QueryPlan:
    """
    Immutable plan produced by the QueryPlanner for a single user query.

    Fields
    ------
    original_query  : The raw query string as received.
    cleaned_query   : Normalised query (whitespace, unicode).
    intent          : Detected QueryIntent.
    strategy        : Retrieval parameters derived from the intent.
    signals         : Human-readable list of detection signals (for logging/debug).
    requires_sql    : True if a SQL agent should be invoked instead of/in addition
                      to vector retrieval.
    multilingual    : True if the query contains non-ASCII characters or
                      detected non-English tokens.
    """
    original_query: str
    cleaned_query: str
    intent: QueryIntent
    strategy: RetrievalStrategy
    signals: List[str] = field(default_factory=list)
    requires_sql: bool = False
    multilingual: bool = False

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "cleaned_query": self.cleaned_query,
            "intent": self.intent.value,
            "signals": self.signals,
            "requires_sql": self.requires_sql,
            "multilingual": self.multilingual,
            "strategy": {
                "top_k": self.strategy.top_k,
                "alpha": self.strategy.alpha,
                "use_reranker": self.strategy.use_reranker,
                "chunk_types": self.strategy.chunk_types,
                "use_sql_agent": self.strategy.use_sql_agent,
            },
        }


# ── Intent Detection Rules ────────────────────────────────────────────────────

# Patterns are applied in order; first match wins (except MULTILINGUAL which
# can stack with another intent).

_FACTUAL_PATTERNS = re.compile(
    r"\b(what\s+is|what\s+are|define|definition\s+of|explain|describe|who\s+is"
    r"|tell\s+me\s+about|meaning\s+of|overview\s+of)\b",
    re.IGNORECASE,
)

_ANALYTICAL_PATTERNS = re.compile(
    r"\b(compare|comparison|difference|versus|vs\.?|pros\s+and\s+cons|advantages"
    r"|disadvantages|analyse|analyze|evaluate|assess|contrast|trade.?off)\b",
    re.IGNORECASE,
)

_NUMERICAL_PATTERNS = re.compile(
    r"\b(how\s+much|how\s+many|calculate|percentage|ratio"
    r"|average|median|sum|total|count|statistics)\b|\d{1,3}[%$€£¥]|\$\d",
    re.IGNORECASE,
)

_TABULAR_PATTERNS = re.compile(
    r"\b(spreadsheet|excel|csv|show.*table|data.*table|table.*data)\b",
    re.IGNORECASE,
)

_SQL_PATTERNS = re.compile(
    r"\b(select.*from|group\s+by|order\s+by|sql\s+query|run\s+sql|execute\s+query)\b",
    re.IGNORECASE,
)

_SUMMARISATION_PATTERNS = re.compile(
    r"\b(summarize|summarise|summary|overview\s+of\s+the\s+(document|paper|report)"
    r"|key\s+points|main\s+ideas|brief\s+description|tl;?dr|in\s+short"
    r"|highlight\s+the|important\s+points)\b",
    re.IGNORECASE,
)

_IMAGE_PATTERNS = re.compile(
    r"\b(figure|fig\.|diagram|chart|graph|image|illustration|caption)\b",
    re.IGNORECASE,
)


def _is_multilingual(text: str) -> bool:
    """
    Detect non-ASCII / non-Latin scripts that suggest multilingual content.
    Returns True if > 5% of characters are outside Basic Latin.
    """
    if not text:
        return False
    non_latin = sum(
        1 for ch in text
        if unicodedata.category(ch) not in ("Ll", "Lu", "Lt", "Lo", "Nd", "Zs", "Po", "Ps", "Pe")
        and ord(ch) > 127
    )
    return (non_latin / len(text)) > 0.05


def _strategy_for_intent(intent: QueryIntent) -> RetrievalStrategy:
    """Return a RetrievalStrategy tuned for the given intent."""
    mapping: dict[QueryIntent, RetrievalStrategy] = {
        QueryIntent.FACTUAL: RetrievalStrategy(top_k=5, alpha=0.7, use_reranker=True),
        QueryIntent.ANALYTICAL: RetrievalStrategy(top_k=8, alpha=0.6, use_reranker=True),
        QueryIntent.NUMERICAL: RetrievalStrategy(
            top_k=5, alpha=0.5, use_reranker=True,
            chunk_types=None,
        ),
        QueryIntent.TABULAR: RetrievalStrategy(
            top_k=8, alpha=0.4, use_reranker=True,
            chunk_types=["table", "text"],
        ),
        QueryIntent.SQL_DATA: RetrievalStrategy(
            top_k=5, alpha=0.7, use_reranker=True,
            use_sql_agent=False,
        ),
        QueryIntent.MULTILINGUAL: RetrievalStrategy(
            top_k=5, alpha=0.85, use_reranker=True,
        ),
        QueryIntent.SUMMARISATION: RetrievalStrategy(
            top_k=15, alpha=0.65, use_reranker=False,  # reranker less useful for broad recall
        ),
        QueryIntent.IMAGE: RetrievalStrategy(
            top_k=5, alpha=0.6, use_reranker=True,
            chunk_types=None,
        ),
        QueryIntent.GENERAL: RetrievalStrategy(top_k=5, alpha=0.7, use_reranker=True),
    }
    return mapping.get(intent, RetrievalStrategy())


# ── Query Planner ─────────────────────────────────────────────────────────────

class QueryPlanner:
    """
    Phase 6: Query Planner & Intent Router.

    Analyses a raw query string and produces a QueryPlan in <5 ms using
    pure regex/heuristic rules — no LLM, no network, no DB calls.

    The resulting QueryPlan is consumed by the SearchRouter to select
    the appropriate retrieval path.
    """

    def plan(self, query: str) -> QueryPlan:
        """
        Classify the query and build a retrieval plan.

        Parameters
        ----------
        query : str
            The raw user query.

        Returns
        -------
        QueryPlan
            Contains the detected intent, retrieval strategy, and debug signals.
        """
        # Normalise
        cleaned = query.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)

        signals: list[str] = []

        # ── Multilingual detection (always checked) ───────────────────────
        multilingual = _is_multilingual(cleaned)
        if multilingual:
            signals.append("multilingual_chars_detected")

        # ── Primary intent detection (order matters — most specific first) ─
        intent = QueryIntent.GENERAL

        if _SQL_PATTERNS.search(cleaned):
            intent = QueryIntent.SQL_DATA
            signals.append("sql_keywords")

        elif _TABULAR_PATTERNS.search(cleaned):
            intent = QueryIntent.TABULAR
            signals.append("tabular_keywords")

        elif _NUMERICAL_PATTERNS.search(cleaned):
            intent = QueryIntent.NUMERICAL
            signals.append("numerical_keywords")

        elif _IMAGE_PATTERNS.search(cleaned):
            intent = QueryIntent.IMAGE
            signals.append("image_keywords")

        elif _SUMMARISATION_PATTERNS.search(cleaned):
            intent = QueryIntent.SUMMARISATION
            signals.append("summarisation_keywords")

        elif _ANALYTICAL_PATTERNS.search(cleaned):
            intent = QueryIntent.ANALYTICAL
            signals.append("analytical_keywords")

        elif _FACTUAL_PATTERNS.search(cleaned):
            intent = QueryIntent.FACTUAL
            signals.append("factual_keywords")

        # Override to MULTILINGUAL if non-Latin scripts dominate AND no other
        # specific intent was detected. If intent was already detected, we
        # only adjust the alpha (handled in strategy).
        if multilingual and intent == QueryIntent.GENERAL:
            intent = QueryIntent.MULTILINGUAL

        strategy = _strategy_for_intent(intent)

        # For multilingual, boost alpha regardless of other intent
        if multilingual and intent != QueryIntent.MULTILINGUAL:
            strategy.alpha = min(strategy.alpha + 0.1, 0.95)
            signals.append("alpha_boosted_for_multilingual")

        plan = QueryPlan(
            original_query=query,
            cleaned_query=cleaned,
            intent=intent,
            strategy=strategy,
            signals=signals,
            requires_sql=(intent == QueryIntent.SQL_DATA),
            multilingual=multilingual,
        )

        logger.info(
            "Query plan created",
            intent=intent.value,
            signals=signals,
            top_k=strategy.top_k,
            alpha=strategy.alpha,
        )
        return plan
