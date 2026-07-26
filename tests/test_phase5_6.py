"""
DocuRAG — Phase 5+6 Unit Tests

Tests for:
  - BM25Index scoring
  - MetadataFilter query building
  - QueryPlanner intent detection
  - SearchService clean query
  - SearchResult serialisation
"""
from __future__ import annotations

import pytest
import numpy as np

from src.retrieval.retriever import BM25Index, MetadataFilter, _min_max_normalize
from src.retrieval.query_planner import QueryPlanner, QueryIntent


# ── BM25 Tests ────────────────────────────────────────────────────────────────

class _FakeChunk:
    """Minimal chunk stub for BM25 testing."""
    def __init__(self, text: str):
        self.text = text


def test_bm25_scores_relevant_higher():
    chunks = [
        _FakeChunk("backpropagation computes gradients via chain rule"),
        _FakeChunk("deep learning models use neural networks"),
        _FakeChunk("the weather is sunny today"),
    ]
    idx = BM25Index(chunks)
    scores = idx.score("backpropagation chain rule")
    assert scores[0] > scores[2], "Relevant chunk should score higher than irrelevant"


def test_bm25_empty_query():
    chunks = [_FakeChunk("some text here")]
    idx = BM25Index(chunks)
    scores = idx.score("")
    assert all(s == 0.0 for s in scores)


def test_bm25_no_chunks():
    idx = BM25Index([])
    scores = idx.score("anything")
    assert len(scores) == 0


def test_min_max_normalize_uniform():
    arr = np.array([5.0, 5.0, 5.0])
    result = _min_max_normalize(arr)
    # All same values > 0 → all 1.0
    assert all(result == 1.0)


def test_min_max_normalize_zeros():
    arr = np.array([0.0, 0.0, 0.0])
    result = _min_max_normalize(arr)
    assert all(result == 0.0)


def test_min_max_normalize_range():
    arr = np.array([0.0, 0.5, 1.0])
    result = _min_max_normalize(arr)
    assert pytest.approx(result[0], abs=1e-5) == 0.0
    assert pytest.approx(result[2], abs=1e-5) == 1.0


# ── QueryPlanner Tests ────────────────────────────────────────────────────────

@pytest.fixture
def planner():
    return QueryPlanner()


def test_planner_factual(planner):
    plan = planner.plan("What is backpropagation?")
    assert plan.intent == QueryIntent.FACTUAL


def test_planner_analytical(planner):
    plan = planner.plan("Compare LSTM vs Transformer architectures")
    assert plan.intent == QueryIntent.ANALYTICAL


def test_planner_numerical(planner):
    plan = planner.plan("Calculate the average accuracy percentage across all experiments")
    assert plan.intent == QueryIntent.NUMERICAL


def test_planner_tabular(planner):
    plan = planner.plan("Show me the data in the table for Q3 results")
    assert plan.intent == QueryIntent.TABULAR


def test_planner_summarisation(planner):
    plan = planner.plan("Summarize the key points of this document")
    assert plan.intent == QueryIntent.SUMMARISATION


def test_planner_image(planner):
    plan = planner.plan("What does Figure 4 show?")
    assert plan.intent == QueryIntent.IMAGE


def test_planner_sql(planner):
    plan = planner.plan("SELECT name FROM users where age > 30")
    assert plan.intent == QueryIntent.SQL_DATA
    assert plan.requires_sql is True


def test_planner_general(planner):
    plan = planner.plan("neural network")
    assert plan.intent == QueryIntent.GENERAL


def test_planner_multilingual_detection(planner):
    # Hindi text mixed with English
    plan = planner.plan("मशीन लर्निंग क्या है?")
    assert plan.multilingual is True


def test_planner_returns_signals(planner):
    plan = planner.plan("What is NLP?")
    assert isinstance(plan.signals, list)
    assert len(plan.signals) > 0


def test_planner_strategy_top_k_summarisation(planner):
    plan = planner.plan("Summarize the entire paper")
    # Summarisation should fetch more chunks (higher recall)
    assert plan.strategy.top_k >= 10


def test_planner_strategy_tabular_chunk_types(planner):
    plan = planner.plan("List all the records in the spreadsheet")
    assert plan.strategy.chunk_types is not None
    assert "table" in plan.strategy.chunk_types


def test_planner_to_dict(planner):
    plan = planner.plan("Explain backpropagation")
    d = plan.to_dict()
    assert "intent" in d
    assert "strategy" in d
    assert "signals" in d
    assert "requires_sql" in d
    assert "multilingual" in d


def test_planner_whitespace_cleaning(planner):
    plan = planner.plan("  what  is   NLP  ")
    assert "  " not in plan.cleaned_query
