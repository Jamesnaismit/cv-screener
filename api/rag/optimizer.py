"""
Prompt optimizer orchestration.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .prompt_template import PromptTemplate
from .query_analyzer import QueryAnalyzer
from .guardrails import GuardrailValidator

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """
    Main class for optimized prompt generation with semantic enhancements.

    Integrates:
    - Query analysis (language, complexity)
    - Context-aware augmentation
    - Hierarchical prompt structuring
    - Enhanced validation
    """

    def __init__(
            self,
            use_few_shot: bool = True,
            validate_output: bool = True,
            auto_augment_short_queries: bool = True,
            openai_client=None,
    ):
        """
        Initialize prompt optimizer.

        Args:
            use_few_shot: Whether to include few-shot examples.
            validate_output: Whether to validate outputs.
            auto_augment_short_queries: Whether to auto-augment short queries.
            openai_client: Optional OpenAI client for LLM-based language detection (not used; kept for interface).
        """
        self.use_few_shot = use_few_shot
        self.validate_output = validate_output
        self.auto_augment_short_queries = auto_augment_short_queries
        self.validator = GuardrailValidator()
        self.analyzer = QueryAnalyzer(openai_client=openai_client)

        logger.info(
            f"PromptOptimizer initialized: few_shot={use_few_shot}, "
            f"validate={validate_output}, auto_augment={auto_augment_short_queries}, "
            f"language_mode=english_only"
        )

    def create_prompt(
            self,
            context: str,
            chat_history: str,
            question: str,
            chat_history_list: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Create optimized prompt with automatic analysis and adaptation.

        Args:
            context: Retrieved context (should be pre-formatted with quality tiers).
            chat_history: Formatted conversation history string.
            question: User question.
            chat_history_list: Optional list of (question, answer) tuples for augmentation.

        Returns:
            Tuple of (optimized_prompt, analysis_metadata)
        """
        language = self.analyzer.detect_language(question)
        complexity = self.analyzer.classify_complexity(question)

        original_question = question
        was_augmented = False
        if self.auto_augment_short_queries and chat_history_list:
            question, was_augmented = self.analyzer.augment_short_query(
                question, chat_history_list
            )

        prompt = PromptTemplate.create_full_prompt(
            context=context,
            chat_history=chat_history,
            question=question,
            language=language,
            query_complexity=complexity,
            include_few_shot=self.use_few_shot,
        )

        metadata = {
            "language": language,
            "complexity": complexity,
            "original_question": original_question,
            "augmented_question": question if was_augmented else None,
            "was_augmented": was_augmented,
            "prompt_length": len(prompt),
        }

        logger.info(
            f"Created prompt: lang={language}, complexity={complexity}, "
            f"augmented={was_augmented}, length={len(prompt)}"
        )

        return prompt, metadata

    def validate_response(
            self,
            response: str,
            context: str,
            sources: List[Dict],
            language: str = "en",
    ) -> Dict[str, Any]:
        """
        Validate generated response with enhanced checks.

        Args:
            response: Generated response.
            context: Context used.
            sources: Source documents.
            language: Expected language.

        Returns:
            Comprehensive validation results.
        """
        if not self.validate_output:
            return {
                "passed": True,
                "issues": [],
                "score": 1.0,
                "validation_skipped": True
            }

        validation = self.validator.validate_response(
            response, context, sources, language
        )

        fabricated_claims = self.validator.detect_fabricated_claims(response, language)

        result = {
            **validation,
            "fabricated_claims": fabricated_claims,
            "has_fabricated_claims": len(fabricated_claims) > 0,
        }

        if fabricated_claims:
            result["issues"].append(f"Potential fabricated claims: {fabricated_claims}")
            result["passed"] = False
            result["score"] = max(0.0, result["score"] - 0.3)

        logger.info(
            f"Validation: passed={result['passed']}, score={result['score']:.2f}, "
            f"issues={len(result['issues'])}"
        )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics and configuration."""
        return {
            "use_few_shot": self.use_few_shot,
            "validate_output": self.validate_output,
            "auto_augment_short_queries": self.auto_augment_short_queries,
            "version": "2.0",
            "features": [
                "english_only",
                "chain_of_thought",
                "query_augmentation",
                "tiered_context",
                "enhanced_validation",
                "confidence_expression",
            ]
        }
