"""
DocuRAG — Legal & Policy Specialist Agent

Specialized in government guidelines, Gazette notifications, Acts, and PGLM clauses.
Extracts exact clause numbers, section titles, and statutory references to boost
retrieval accuracy for complex policy queries.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)

# Legal and Policy Clause Matching Regex Patterns
CLAUSE_PATTERNS = [
    r"\b(?:section|para|paragraph|clause|article|issue)\s+(\d+(?:\.\d+)*(?:\([a-z0-9]+\))*)",
    r"\b(?:act|notification|guideline|rules)\s+(?:of\s+)?(\d{4})",
    r"\b([A-Z]{2,6}\s+\d{4})",
]


class LegalPolicyAgent:
    """Specialized agent for processing legal, policy, and statutory document queries."""

    def __init__(self):
        pass

    def analyze_legal_query(self, query: str) -> Dict[str, Optional[List[str]]]:
        """
        Analyze a query to extract specific legal clauses, act names, or section numbers.
        """
        clauses = []
        for pattern in CLAUSE_PATTERNS:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for m in matches:
                if isinstance(m, tuple):
                    m = m[0]
                if m and m not in clauses:
                    clauses.append(m)

        is_legal = bool(
            clauses
            or re.search(
                r"\b(act|law|policy|guideline|gazette|sor|pglm|mpt|mrtp|clause|section|tenure|lease|concession|nomination|tender)\b",
                query,
                re.IGNORECASE,
            )
        )

        logger.info("LegalPolicyAgent analysis complete", is_legal=is_legal, clauses=clauses)
        return {
            "is_legal": is_legal,
            "extracted_clauses": clauses,
        }
