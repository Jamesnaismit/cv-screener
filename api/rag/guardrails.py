"""
Guardrail validation for generated responses.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class GuardrailValidator:
    """Validate responses for citations, language, and hallucination cues."""

    @staticmethod
    def validate_response(
            response: str,
            context: str,
            sources: List[Dict],
            language: str = "en"
    ) -> Dict[str, Any]:
        """
        Validate a response against enhanced guardrails.

        Args:
            response: Generated response.
            context: Context used for generation.
            sources: Source documents.
            language: Expected response language.

        Returns:
            Validation result with pass/fail and detailed issues.
        """
        issues = []

        if not response or len(response.strip()) < 10:
            issues.append("Response too short or empty")

        inline_citations = re.findall(r'\[\d+\]', response)
        if sources and not inline_citations:
            issues.append("Missing inline citations [N]")

        footnote_markers = ["**Sources consulted:**"]
        has_footnotes = any(marker in response for marker in footnote_markers)
        if sources and not has_footnotes:
            issues.append("Missing footnote section with sources")

        if sources and inline_citations:
            cited_numbers = set(int(re.search(r'\d+', c).group()) for c in inline_citations)
            available_numbers = set(range(1, len(sources) + 1))
            unused_sources = available_numbers - cited_numbers
            if unused_sources:
                issues.append(f"Sources not cited: {unused_sources}")

            phantom = cited_numbers - available_numbers
            if phantom:
                issues.append(f"Citations to non-existent sources: {phantom}")

        if context and len(context) > 50:
            common_words = set(context.lower().split()) & set(response.lower().split())
            if len(common_words) / max(len(set(response.lower().split())), 1) > 0.8:
                issues.append("Response appears to copy context without synthesis")

        expected_words = {
            "en": ["the", "is", "are", "of", "and", "to", "a", "in", "for", "with"]
        }

        words = response.lower().split()
        target_words = expected_words.get(language, expected_words["en"])
        lang_ratio = sum(1 for w in words if w in target_words) / max(len(words), 1)

        if lang_ratio < 0.08:
            issues.append(f"Response may not be in expected language ({language})")

        word_count = len(response.split())
        if word_count > 700:
            issues.append(f"Response too long ({word_count} words, max 600)")

        hallucination_phrases = {
            "en": ["in my knowledge", "as far as I know", "in my experience",
                   "I think", "I imagine", "probably without"]
        }

        response_lower = response.lower()
        found_hallucinations = []
        for phrase in hallucination_phrases.get(language, hallucination_phrases["en"]):
            if phrase in response_lower:
                found_hallucinations.append(phrase)

        if found_hallucinations:
            issues.append(f"Hallucination indicators found: {found_hallucinations}")

        score = max(0.0, 1.0 - (len(issues) * 0.15))

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "score": score,
            "inline_citations_count": len(inline_citations),
            "has_footnotes": has_footnotes,
        }

    @staticmethod
    def detect_fabricated_claims(response: str, language: str = "en") -> List[str]:
        """
        Detect potentially fabricated specific claims.

        Args:
            response: Generated response.
            language: Response language.

        Returns:
            List of suspicious claims found.
        """
        suspicious_claims = []

        claim_markers = {
            "en": [
                (r'\d+[\s]*€', "specific price in euros"),
                (r'\d+[\s]*\$', "specific price in dollars"),
                (r'founded in \d{4}', "founding year"),
                (r'costs.*\d+', "cost statement"),
                (r'price.*\d+', "price statement"),
                (r'\d+ employees', "employee count"),
                (r'\d+% of', "specific percentage"),
            ]
        }

        response_lower = response.lower()
        markers = claim_markers.get(language, claim_markers["en"])

        for pattern, claim_type in markers:
            matches = re.findall(pattern, response_lower)
            if matches:
                suspicious_claims.append(f"{claim_type}: {matches}")

        return suspicious_claims
