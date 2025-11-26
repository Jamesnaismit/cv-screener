"""
Query analysis helpers (language detection, complexity, augmentation).
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Simple analyzer for query language and complexity."""

    def __init__(self, openai_client=None):
        """
        Initialize QueryAnalyzer

        Args:
            openai_client: Optional OpenAI client (kept for interface compatibility).
        """
        self.openai_client = openai_client

    def detect_language(self, query: str) -> str:
        """
        Detect query language

        Args:
            query: User query

        Returns:
            Language code: always "en"
        """
        return "en"

    @staticmethod
    def classify_complexity(query: str) -> str:
        """
        Classify query complexity to determine processing strategy.

        Args:
            query: User query

        Returns:
            Complexity level: "simple", "moderate", or "complex"
        """
        query_lower = query.lower()

        complex_markers = [
            "compare", "difference", "pros and cons",
            "why", "how does", "process", "implement",
            "better than", "vs", "versus", "explain"
        ]

        moderate_markers = [
            "features", "capabilities", "includes", "offers",
            "types of", "which"
        ]

        for marker in complex_markers:
            if marker in query_lower:
                return "complex"

        for marker in moderate_markers:
            if marker in query_lower:
                return "moderate"

        return "simple"

    @staticmethod
    def augment_short_query(
            query: str,
            chat_history: List[Tuple[str, str]]
    ) -> Tuple[str, bool]:
        """
        Augment short or ambiguous queries with context from history.

        Args:
            query: User query
            chat_history: Recent conversation history

        Returns:
            Tuple of (augmented_query, was_augmented)
        """
        words = query.split()
        if len(words) > 2:
            return query, False

        if not chat_history:
            return query, False

        last_q, last_a = chat_history[-1]

        topics = {
            "profile": "about the candidate profile",
            "experience": "about the professional experience",
            "skills": "about the candidate skills",
            "education": "about the education",
        }

        for topic, augmentation in topics.items():
            if topic.lower() in last_a.lower():
                augmented = f"{query} {augmentation}"
                logger.info(f"Augmented short query: '{query}' → '{augmented}'")
                return augmented, True

        return query, False
