"""
DocuRAG — Agent Orchestrator

Central coordinator for the Multi-Agent RAG system.
Delegates incoming user queries to specialized sub-agents:
- LegalPolicyAgent (Government guidelines, Gazette notifications, Acts, PGLM clauses)
- MathFormulaAgent (Rate multipliers, EMD 10%, financial calculations)

Combines document evidence and verified mathematical proofs into the final ChatML prompt.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.retrieval.agents.legal_policy_agent import LegalPolicyAgent
from src.retrieval.agents.math_formula_agent import MathFormulaAgent
from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)


class AgentOrchestrator:
    """Orchestrates multi-agent execution for legal, policy, and mathematical queries."""

    def __init__(self):
        self.legal_agent = LegalPolicyAgent()
        self.math_agent = MathFormulaAgent()

    async def execute_rag_pipeline(
        self,
        query: str,
        retriever_func: Any,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Orchestrate multi-agent RAG workflow:
        1. Analyze intent with LegalPolicyAgent & MathFormulaAgent
        2. Perform hybrid retrieval
        3. Evaluate mathematical proofs if math intent is present
        4. Assemble enriched context
        """
        logger.info("AgentOrchestrator initiating pipeline", query=query[:100])

        # Step 1: Legal / Policy intent analysis
        legal_info = self.legal_agent.analyze_legal_query(query)

        # Step 2: Retrieve candidate chunks via retriever function
        retrieved_results = await retriever_func(query=query, document_ids=document_ids, top_k=top_k)

        chunks = [res[1] for res in retrieved_results] if isinstance(retrieved_results, list) and len(retrieved_results) > 0 and isinstance(retrieved_results[0], tuple) else retrieved_results

        # Step 3: Mathematical intent analysis & proof generation
        has_math_intent = self.math_agent.detect_math_intent(query)
        math_info = {"has_math": False, "proof_text": "", "calculations": []}
        if has_math_intent:
            math_info = self.math_agent.process_math_query(query, chunks)

        # Step 4: Assemble enriched prompt instructions
        agent_context_enrichments = []
        if legal_info["is_legal"] and legal_info["extracted_clauses"]:
            agent_context_enrichments.append(
                f"[Agent Insight: Query targets specific clauses: {', '.join(legal_info['extracted_clauses'])}]"
            )

        if math_info["has_math"]:
            agent_context_enrichments.append(
                f"[Mathematical Proof & Verification]:\n{math_info['proof_text']}"
            )

        enrichment_text = "\n".join(agent_context_enrichments)

        return {
            "query": query,
            "chunks": chunks,
            "legal_info": legal_info,
            "math_info": math_info,
            "enrichment_text": enrichment_text,
            "agents_triggered": [
                name for name, active in [
                    ("LegalPolicyAgent", legal_info["is_legal"]),
                    ("MathFormulaAgent", math_info["has_math"]),
                ] if active
            ]
        }
