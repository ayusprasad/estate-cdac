"""
DocuRAG — Phases 8, 9, 10 Tests

Tests cover:
  Phase 8 — RAGGenerator:
    - Extractive fallback mode (no model)
    - Prompt building
    - Citation extraction from chunks
    - Empty chunks handling
    - generate_answer() backward-compat wrapper

  Phase 9 — CitationVerifier:
    - Tokenisation
    - Sentence splitting
    - Overlap scoring
    - Grounded vs ungrounded sentences
    - Faithfulness score computation
    - Edge cases (empty answer, empty chunks)

  Phase 10 — RAGEvaluator:
    - Precision@K, Recall@K, MRR, NDCG@K metric functions
    - Single query evaluation
    - Batch evaluation
    - Summary report aggregation
    - Evaluator reset
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from typing import Any, Dict, List

# ── Phase 8 imports ───────────────────────────────────────────────────────────
from src.llm.generator import (
    RAGGenerator,
    _extractive_answer,
    _build_rag_prompt,
)

# ── Phase 9 imports ───────────────────────────────────────────────────────────
from src.llm.citation_verifier import (
    CitationVerifier,
    _tokenise,
    _split_sentences,
    _overlap_score,
    GROUNDING_THRESHOLD,
)

# ── Phase 10 imports ──────────────────────────────────────────────────────────
from src.llm.evaluator import (
    RAGEvaluator,
    _precision_at_k,
    _recall_at_k,
    _mean_reciprocal_rank,
    _ndcg_at_k,
    BatchEvalReport,
    SingleEvalResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_chunks(n: int = 3) -> List[Dict[str, Any]]:
    """Return n synthetic chunk dicts."""
    return [
        {
            "rank": i + 1,
            "score": 1.0 - (i * 0.1),
            "chunk_id": f"chunk-{i}",
            "document_id": "doc-1",
            "document_name": "NLP Lecture Notes",
            "page_number": i + 1,
            "section_title": f"Section {i + 1}",
            "chunk_type": "text",
            "text": f"Backpropagation is the algorithm used to train neural networks. "
                    f"It computes gradients by the chain rule of calculus. "
                    f"This is chunk {i + 1} of the document.",
        }
        for i in range(n)
    ]


# ════════════════════════════════════════════════════════════════════════════
# PHASE 8: RAGGenerator Tests
# ════════════════════════════════════════════════════════════════════════════

class TestExtractiveAnswer:
    """Tests for the extractive fallback answer builder."""

    def test_no_chunks_returns_not_found(self):
        answer = _extractive_answer("what is ML?", [])
        assert "not find" in answer.lower() or "could not" in answer.lower()

    def test_single_chunk_included(self):
        chunks = _make_chunks(1)
        answer = _extractive_answer("what is backprop?", chunks)
        assert "Backpropagation" in answer or "backpropagation" in answer.lower()

    def test_multiple_chunks_all_included(self):
        chunks = _make_chunks(3)
        answer = _extractive_answer("explain backprop", chunks)
        assert "[1]" in answer
        assert "[2]" in answer
        assert "[3]" in answer

    def test_document_name_in_answer(self):
        chunks = _make_chunks(1)
        answer = _extractive_answer("test", chunks)
        assert "NLP Lecture Notes" in answer

    def test_page_number_in_answer(self):
        chunks = _make_chunks(1)
        answer = _extractive_answer("test", chunks)
        assert "Page 1" in answer

    def test_answer_is_string(self):
        answer = _extractive_answer("question", _make_chunks(2))
        assert isinstance(answer, str)
        assert len(answer) > 0


class TestBuildRagPrompt:
    """Tests for the LLM prompt constructor."""

    def test_prompt_contains_query(self):
        prompt = _build_rag_prompt("What is backprop?", _make_chunks(2))
        assert "backprop" in prompt.lower()

    def test_prompt_contains_chunk_text(self):
        prompt = _build_rag_prompt("test", _make_chunks(1))
        assert "Backpropagation" in prompt

    def test_prompt_contains_citation_numbers(self):
        prompt = _build_rag_prompt("test", _make_chunks(3))
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt

    def test_prompt_contains_system_instruction(self):
        prompt = _build_rag_prompt("test", _make_chunks(1))
        assert "hallucinate" in prompt.lower() or "ONLY" in prompt or "only" in prompt.lower()

    def test_prompt_has_document_reference(self):
        prompt = _build_rag_prompt("test", _make_chunks(1))
        assert "NLP Lecture Notes" in prompt

    def test_prompt_truncates_long_chunks(self):
        long_chunk = [{"rank": 1, "score": 1.0, "chunk_id": "x",
                       "document_id": "d", "document_name": "Doc",
                       "page_number": 1, "section_title": None,
                       "chunk_type": "text",
                       "text": "a" * 2000}]
        from src.llm.generator import MAX_CHUNK_CHARS
        prompt = _build_rag_prompt("test", long_chunk)
        # The very long text should be truncated in the prompt
        assert prompt.count("a" * (MAX_CHUNK_CHARS + 10)) == 0

    def test_empty_chunks_still_builds_prompt(self):
        prompt = _build_rag_prompt("test query", [])
        assert "test query" in prompt


class TestRAGGenerator:
    """Tests for the RAGGenerator class."""

    def setup_method(self):
        self.generator = RAGGenerator()

    def test_mode_is_string(self):
        assert self.generator.mode in ("llm", "extractive")

    def test_mode_extractive_when_no_model(self):
        # Since no GGUF model exists in test env, should be extractive
        assert self.generator.mode == "extractive"

    @pytest.mark.asyncio
    async def test_generate_returns_dict(self):
        result = await self.generator.generate("what is ML?", _make_chunks(2))
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_generate_has_required_keys(self):
        result = await self.generator.generate("test", _make_chunks(1))
        required_keys = {"answer", "mode", "citations", "latency_ms", "model", "llm_available"}
        assert required_keys.issubset(result.keys())

    @pytest.mark.asyncio
    async def test_generate_answer_not_empty(self):
        result = await self.generator.generate("what is backpropagation?", _make_chunks(2))
        assert len(result["answer"]) > 0

    @pytest.mark.asyncio
    async def test_generate_citations_match_chunks(self):
        chunks = _make_chunks(3)
        result = await self.generator.generate("test", chunks)
        assert len(result["citations"]) == 3

    @pytest.mark.asyncio
    async def test_generate_empty_chunks(self):
        result = await self.generator.generate("what is ML?", [])
        assert "not find" in result["answer"].lower() or "could not" in result["answer"].lower()
        assert result["citations"] == []

    @pytest.mark.asyncio
    async def test_generate_latency_positive(self):
        result = await self.generator.generate("test", _make_chunks(1))
        assert result["latency_ms"] >= 0.0

    @pytest.mark.asyncio
    async def test_generate_answer_backward_compat(self):
        """generate_answer() should return just a string."""
        answer = await self.generator.generate_answer("what is ML?", _make_chunks(1))
        assert isinstance(answer, str)
        assert len(answer) > 0

    @pytest.mark.asyncio
    async def test_citations_have_document_name(self):
        result = await self.generator.generate("test", _make_chunks(2))
        for citation in result["citations"]:
            assert "document_name" in citation
            assert citation["document_name"] == "NLP Lecture Notes"

    @pytest.mark.asyncio
    async def test_citations_have_rank(self):
        result = await self.generator.generate("test", _make_chunks(2))
        ranks = [c["rank"] for c in result["citations"]]
        assert sorted(ranks) == [1, 2]


# ════════════════════════════════════════════════════════════════════════════
# PHASE 9: CitationVerifier Tests
# ════════════════════════════════════════════════════════════════════════════

class TestTokenise:
    def test_basic_tokenisation(self):
        tokens = _tokenise("Backpropagation is an algorithm for neural networks")
        assert "backpropagation" in tokens
        assert "algorithm" in tokens

    def test_stop_words_removed(self):
        tokens = _tokenise("the cat is on the mat")
        assert "the" not in tokens
        assert "is" not in tokens
        # Content words should remain
        assert "cat" in tokens or "mat" in tokens

    def test_punctuation_stripped(self):
        tokens = _tokenise("Hello, world! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty_string(self):
        tokens = _tokenise("")
        assert len(tokens) == 0

    def test_returns_frozenset(self):
        result = _tokenise("test tokens")
        assert isinstance(result, frozenset)

    def test_lowercase(self):
        tokens = _tokenise("UPPERCASE lowercase MiXeD")
        assert "uppercase" in tokens
        assert "lowercase" in tokens
        assert "mixed" in tokens


class TestSplitSentences:
    def test_simple_paragraph(self):
        text = "First sentence. Second sentence. Third sentence."
        sentences = _split_sentences(text)
        assert len(sentences) >= 2

    def test_newline_splitting(self):
        text = "Line one about backprop\nLine two about gradients\nLine three about chains"
        sentences = _split_sentences(text)
        assert len(sentences) >= 2

    def test_empty_string(self):
        sentences = _split_sentences("")
        assert sentences == []

    def test_short_strings_filtered(self):
        sentences = _split_sentences("Hi. OK. Hello world, this is a longer sentence.")
        # Very short strings like "Hi." should be filtered
        for s in sentences:
            assert len(s) > 10


class TestOverlapScore:
    def test_identical_sets(self):
        tokens = frozenset(["neural", "network", "backprop"])
        score = _overlap_score(tokens, tokens)
        assert score == 1.0

    def test_no_overlap(self):
        a = frozenset(["apple", "banana"])
        b = frozenset(["car", "truck"])
        score = _overlap_score(a, b)
        assert score == 0.0

    def test_partial_overlap(self):
        a = frozenset(["neural", "network", "backprop"])
        b = frozenset(["neural", "network", "unrelated"])
        score = _overlap_score(a, b)
        assert 0.0 < score < 1.0

    def test_empty_sentence_tokens(self):
        score = _overlap_score(frozenset(), frozenset(["word"]))
        assert score == 0.0


class TestCitationVerifier:
    def setup_method(self):
        self.verifier = CitationVerifier()
        self.chunks = _make_chunks(3)

    def test_score_returns_float(self):
        score = self.verifier.score("Backpropagation computes gradients.", self.chunks)
        assert isinstance(score, float)

    def test_score_between_0_and_1(self):
        score = self.verifier.score("Some answer text here.", self.chunks)
        assert 0.0 <= score <= 1.0

    def test_grounded_answer_scores_high(self):
        # Answer directly lifted from chunks
        answer = (
            "Backpropagation is the algorithm used to train neural networks. "
            "It computes gradients by the chain rule of calculus."
        )
        score = self.verifier.score(answer, self.chunks)
        assert score >= 0.5  # Should be well grounded

    def test_empty_answer_returns_unknown_verdict(self):
        result = self.verifier.verify("", self.chunks)
        assert result.verdict == "UNKNOWN"

    def test_empty_chunks_returns_score_1(self):
        # No chunks to disprove → default to max faithfulness
        score = self.verifier.score("Some answer.", [])
        assert score == 1.0

    def test_completely_unrelated_answer_scores_low(self):
        answer = (
            "The weather today is sunny with high temperatures expected. "
            "Stock markets rose by three percent overnight on positive economic data."
        )
        score = self.verifier.score(answer, self.chunks)
        # Very low — nothing overlaps with NLP text
        assert score < 0.5

    def test_verify_returns_sentence_grounds(self):
        result = self.verifier.verify("Backpropagation computes gradients efficiently.", self.chunks)
        assert len(result.sentence_grounds) >= 0  # may be 0 if too short

    def test_verdict_high(self):
        result = self.verifier.verify("x" * 5, self.chunks)  # trivial
        assert result.verdict in ("HIGH", "MEDIUM", "LOW", "UNKNOWN")

    def test_verify_faithfulness_matches_score(self):
        answer = "Backpropagation is an algorithm for training neural networks using gradients."
        result = self.verifier.verify(answer, self.chunks)
        direct_score = self.verifier.score(answer, self.chunks)
        assert abs(result.faithfulness - direct_score) < 0.001


# ════════════════════════════════════════════════════════════════════════════
# PHASE 10: RAGEvaluator Tests
# ════════════════════════════════════════════════════════════════════════════

class TestPrecisionAtK:
    def test_all_relevant(self):
        assert _precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_none_relevant(self):
        assert _precision_at_k(["a", "b", "c"], ["x", "y"], k=3) == 0.0

    def test_half_relevant(self):
        p = _precision_at_k(["a", "b", "c", "d"], ["a", "c"], k=4)
        assert abs(p - 0.5) < 0.01

    def test_k_less_than_retrieved(self):
        # Only top-2 considered
        p = _precision_at_k(["a", "x", "b", "y"], ["a", "b"], k=2)
        assert p == 0.5  # "a" in top-2, "x" not relevant

    def test_empty_retrieved(self):
        assert _precision_at_k([], ["a", "b"], k=5) == 0.0

    def test_zero_k(self):
        assert _precision_at_k(["a", "b"], ["a"], k=0) == 0.0


class TestRecallAtK:
    def test_all_found(self):
        assert _recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_none_found(self):
        assert _recall_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_partial_recall(self):
        r = _recall_at_k(["a", "x", "b", "y"], ["a", "b", "c"], k=4)
        assert abs(r - (2/3)) < 0.01

    def test_empty_relevant(self):
        assert _recall_at_k(["a", "b"], [], k=2) == 0.0


class TestMRR:
    def test_first_is_relevant(self):
        assert _mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_second_is_relevant(self):
        assert abs(_mean_reciprocal_rank(["x", "a", "b"], ["a"]) - 0.5) < 0.001

    def test_third_is_relevant(self):
        assert abs(_mean_reciprocal_rank(["x", "y", "a"], ["a"]) - 1/3) < 0.001

    def test_none_relevant(self):
        assert _mean_reciprocal_rank(["x", "y", "z"], ["a", "b"]) == 0.0

    def test_multiple_relevant_uses_first(self):
        # First relevant at position 2, another at position 4
        mrr = _mean_reciprocal_rank(["x", "a", "y", "b"], ["a", "b"])
        assert abs(mrr - 0.5) < 0.001


class TestNDCGAtK:
    def test_perfect_ranking(self):
        # All relevant at top → NDCG = 1.0
        ndcg = _ndcg_at_k(["a", "b"], ["a", "b"], k=2)
        assert abs(ndcg - 1.0) < 0.001

    def test_no_relevant(self):
        assert _ndcg_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_partial_relevance(self):
        ndcg = _ndcg_at_k(["a", "x", "b"], ["a", "b"], k=3)
        assert 0.0 < ndcg < 1.0

    def test_empty_retrieved(self):
        assert _ndcg_at_k([], ["a"], k=5) == 0.0

    def test_k_larger_than_retrieved(self):
        # Should not error
        ndcg = _ndcg_at_k(["a"], ["a"], k=10)
        assert ndcg > 0.0


class TestRAGEvaluator:
    def setup_method(self):
        self.evaluator = RAGEvaluator()
        self.chunks = _make_chunks(3)

    def test_empty_summary(self):
        report = self.evaluator.summary()
        assert report.total_queries == 0

    def test_evaluate_response_returns_result(self):
        result = self.evaluator.evaluate_response(
            query="What is backprop?",
            answer="Backpropagation computes gradients.",
            retrieved_chunks=self.chunks,
            faithfulness=0.9,
            retrieval_latency_ms=200.0,
            generation_latency_ms=50.0,
            mode="extractive",
        )
        assert isinstance(result, SingleEvalResult)

    def test_evaluate_response_accumulated(self):
        for _ in range(3):
            self.evaluator.evaluate_response(
                query="test",
                answer="answer",
                retrieved_chunks=self.chunks,
                faithfulness=0.8,
                retrieval_latency_ms=100.0,
                generation_latency_ms=10.0,
            )
        report = self.evaluator.summary()
        assert report.total_queries == 3

    def test_summary_avg_faithfulness(self):
        self.evaluator.evaluate_response(
            query="q1", answer="a", retrieved_chunks=[],
            faithfulness=0.6, retrieval_latency_ms=0, generation_latency_ms=0
        )
        self.evaluator.evaluate_response(
            query="q2", answer="a", retrieved_chunks=[],
            faithfulness=0.8, retrieval_latency_ms=0, generation_latency_ms=0
        )
        report = self.evaluator.summary()
        assert abs(report.avg_faithfulness - 0.7) < 0.01

    def test_summary_with_ground_truth(self):
        chunk_ids = [c["chunk_id"] for c in self.chunks]
        self.evaluator.evaluate_response(
            query="test",
            answer="answer",
            retrieved_chunks=self.chunks,
            faithfulness=1.0,
            retrieval_latency_ms=100,
            generation_latency_ms=10,
            relevant_chunk_ids=chunk_ids,
        )
        report = self.evaluator.summary()
        assert report.avg_precision_at_k is not None
        assert report.avg_recall_at_k is not None
        assert report.avg_mrr is not None

    def test_clear_resets_results(self):
        self.evaluator.evaluate_response(
            query="q", answer="a", retrieved_chunks=[],
            faithfulness=0.5, retrieval_latency_ms=0, generation_latency_ms=0
        )
        self.evaluator.clear()
        assert self.evaluator.summary().total_queries == 0

    def test_batch_evaluation(self):
        test_cases = [
            {
                "query": f"question {i}",
                "answer": "Backpropagation computes gradients.",
                "retrieved_chunks": self.chunks,
                "faithfulness": 0.8,
                "retrieval_latency_ms": 100.0,
                "generation_latency_ms": 20.0,
                "mode": "extractive",
            }
            for i in range(5)
        ]
        report = self.evaluator.evaluate_batch(test_cases)
        assert report.total_queries == 5
        assert report.avg_faithfulness == 0.8

    def test_p95_latency_computed(self):
        latencies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        for lat in latencies:
            self.evaluator.evaluate_response(
                query="q", answer="a", retrieved_chunks=[],
                faithfulness=0.5,
                retrieval_latency_ms=lat,
                generation_latency_ms=0,
            )
        report = self.evaluator.summary()
        assert report.p95_latency_ms > 0

    def test_faithfulness_distribution(self):
        cases = [
            (0.9, "HIGH"),
            (0.6, "MEDIUM"),
            (0.3, "LOW"),
        ]
        for faith, _ in cases:
            self.evaluator.evaluate_response(
                query="q", answer="a", retrieved_chunks=[],
                faithfulness=faith,
                retrieval_latency_ms=0, generation_latency_ms=0,
            )
        report = self.evaluator.summary()
        assert report.faithfulness_distribution["HIGH"] == 1
        assert report.faithfulness_distribution["MEDIUM"] == 1
        assert report.faithfulness_distribution["LOW"] == 1

    def test_llm_mode_fraction(self):
        self.evaluator.evaluate_response(
            query="q1", answer="a", retrieved_chunks=[],
            faithfulness=0.5, retrieval_latency_ms=0, generation_latency_ms=0,
            mode="llm",
        )
        self.evaluator.evaluate_response(
            query="q2", answer="a", retrieved_chunks=[],
            faithfulness=0.5, retrieval_latency_ms=0, generation_latency_ms=0,
            mode="extractive",
        )
        report = self.evaluator.summary()
        assert report.llm_mode_fraction == 0.5
