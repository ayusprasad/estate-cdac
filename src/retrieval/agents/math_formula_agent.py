"""
DocuRAG — Mathematical & Formula Execution Agent

Detects mathematical formulas, rate calculations (e.g. 1.2 * SoR, 10% EMD, valuation formulas),
evaluates exact arithmetic in Python, and generates step-by-step mathematical proofs.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)


class MathFormulaAgent:
    """Agent specialized in identifying and evaluating mathematical expressions & financial formulas."""

    def __init__(self):
        pass

    def detect_math_intent(self, query: str) -> bool:
        """Check if query contains mathematical or numerical calculation intents."""
        math_keywords = [
            r"\bcalculate\b", r"\bcompute\b", r"\bformula\b", r"\brate\b", r"\bmultiplier\b",
            r"\bpercentage\b", r"\bemd\b", r"\bsor\b", r"\btotal\b", r"\bsum\b", r"\baverage\b",
            r"\b ratio\b", r"\bvaluat(e|ion)\b", r"[\d\.]+\s*[\*\+\/\-\%]", r"\d+%"
        ]
        pattern = "|".join(math_keywords)
        return bool(re.search(pattern, query, re.IGNORECASE))

    def process_math_query(self, query: str, context_chunks: List[Any]) -> Dict[str, Any]:
        """
        Extract numerical constants, evaluate rates, and generate a mathematical proof string.
        """
        logger.info("MathFormulaAgent processing query", query=query[:100])
        calculations = []

        # 1. Look for multiplier patterns (e.g., "1.2 times SoR", "1.2 x SoR", "10% EMD")
        multiplier_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:times|x|\*)\s*(?:of\s+)?(sor|market value|rate)", query, re.IGNORECASE)
        number_match = re.search(r"\b(\d+(?:\,\d+)*(?:\.\d+)?)\b", query)

        # 2. Extract numbers if provided in query
        numbers = [float(n.replace(",", "")) for n in re.findall(r"\b\d+(?:\.\d+)?\b", query)]

        # 3. Check for specific formulas in context (e.g. 1.2 * SoR, 10% EMD)
        sor_value = None
        for chunk in context_chunks:
            text = getattr(chunk, "text", str(chunk))
            # Extract SoR rate if mentioned in chunk
            sor_match = re.search(r"SoR\s*(?:rate|value)?\s*(?:of|=|is|:)\s*(?:Rs\.?|INR)?\s*([\d\.\,]+)", text, re.IGNORECASE)
            if sor_match:
                try:
                    sor_value = float(sor_match.group(1).replace(",", ""))
                    break
                except ValueError:
                    pass

        proof_steps = []
        if multiplier_match and sor_value:
            factor = float(multiplier_match.group(1))
            res = factor * sor_value
            proof_steps.append(f"Formula: Rate = {factor} × SoR")
            proof_steps.append(f"Given SoR Rate = {sor_value:,.2f}")
            proof_steps.append(f"Calculation: {factor} × {sor_value:,.2f} = {res:,.2f}")
            calculations.append({"formula": f"{factor} * SoR", "result": res})

        elif "emd" in query.lower() and sor_value:
            emd_res = 0.10 * sor_value
            proof_steps.append("Formula: EMD = 10% of SoR Rate")
            proof_steps.append(f"Given SoR Rate = {sor_value:,.2f}")
            proof_steps.append(f"Calculation: 0.10 × {sor_value:,.2f} = {emd_res:,.2f}")
            calculations.append({"formula": "10% * SoR", "result": emd_res})

        elif numbers and len(numbers) >= 2 and any(op in query for op in ["*", "+", "-", "/", "times"]):
            if "*" in query or "times" in query:
                res = numbers[0] * numbers[1]
                proof_steps.append(f"Calculation: {numbers[0]} × {numbers[1]} = {res:,.2f}")
                calculations.append({"formula": f"{numbers[0]} * {numbers[1]}", "result": res})
            elif "+" in query:
                res = numbers[0] + numbers[1]
                proof_steps.append(f"Calculation: {numbers[0]} + {numbers[1]} = {res:,.2f}")
                calculations.append({"formula": f"{numbers[0]} + {numbers[1]}", "result": res})

        proof_text = "\n".join(proof_steps) if proof_steps else ""

        return {
            "has_math": bool(proof_steps),
            "proof_text": proof_text,
            "calculations": calculations,
        }
