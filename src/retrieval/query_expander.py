"""
DocuRAG — Query Expansion Module

Generates legal, policy, and domain-specific query variations and synonyms to expand
retrieval recall before running hybrid BM25 + BGE vector search.
"""
from __future__ import annotations

import re
from typing import List

from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)

# Common legal and policy domain synonyms for expansion
DOMAIN_SYNONYMS = {
    "sor": ["Schedule of Rates", "SoR", "market value", "land rate"],
    "pglm": ["Policy Guidelines for Land Management", "PGLM", "land guidelines", "lease guidelines"],
    "ppp": ["Public Private Partnership", "PPP", "concession agreement", "concessionaire"],
    "psu": ["Public Sector Undertaking", "CPSU", "SPSU", "Government body"],
    "lease": ["license", "allotment", "tenure", "rent", "leasehold"],
    "renewal": ["fresh lease", "extension", "grant of lease"],
    "custom bond": ["customs area", "bonded area", "port area"],
    "mpt": ["Major Port Trusts", "MPA", "Port Authority"],
    "mrtp": ["Monopolies and Restrictive Trade Practices", "MRTP"],
}


class QueryExpander:
    """Expands queries using domain synonym maps and rule-based sub-query generation."""

    def __init__(self, max_expansions: int = 3):
        self.max_expansions = max_expansions

    def expand(self, query: str) -> List[str]:
        """
        Given an input query, return a list containing the original query
        plus expanded search term variations.
        """
        if not query or not query.strip():
            return [query]

        expanded_queries = [query.strip()]
        lower_query = query.lower()

        # 1. Expand domain abbreviations & terms
        added_synonyms = []
        for term, synonyms in DOMAIN_SYNONYMS.items():
            if re.search(r"\b" + re.escape(term) + r"\b", lower_query):
                for syn in synonyms:
                    if syn.lower() not in lower_query and syn not in added_synonyms:
                        added_synonyms.append(syn)
                        if len(added_synonyms) >= self.max_expansions:
                            break

        if added_synonyms:
            expanded = f"{query} ({' OR '.join(added_synonyms)})"
            expanded_queries.append(expanded)

        # 2. Add question-to-statement variation if question format
        if lower_query.startswith(("what is", "where is", "how to", "clarification on", "issue regarding")):
            statement_var = re.sub(
                r"^(what is|where is|how to|clarification on|issue regarding)\s+",
                "",
                lower_query,
                flags=re.IGNORECASE,
            ).strip()
            if statement_var and statement_var not in expanded_queries:
                expanded_queries.append(statement_var)

        logger.info("Query expansion generated", original=query, count=len(expanded_queries))
        return expanded_queries
