"""
Conversational RAG chain module.

This module implements the RAG pipeline with conversation memory,
combining retrieval with LLM generation for context-aware responses.
Enhanced with caching, metrics, re-ranking, and prompt optimization.
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

from .cache import ResponseCache
from .metrics import MetricsCollector
from .optimizer import PromptOptimizer

logger = logging.getLogger(__name__)


class ConversationalRAGChain:
    """
    RAG chain with conversation memory for context-aware question answering.

    Enhanced with:
    - Quality-tiered context formatting
    - Multilingual support
    - Chain-of-Thought for complex queries
    - Enhanced validation
    """

    def __init__(
            self,
            retriever,
            openai_api_key: str,
            model_name: str = "gpt-5-mini-2025-08-07",
            max_history: int = 10,
            cache: Optional[ResponseCache] = None,
            metrics: Optional[MetricsCollector] = None,
            prompt_optimizer: Optional[PromptOptimizer] = None,
    ) -> None:
        """
        Initialize the conversational RAG chain.

        Args:
            retriever: VectorRetriever or HybridRetriever instance.
            openai_api_key: OpenAI API key.
            model_name: Name of the LLM model to use.
            max_history: Maximum number of conversation turns to keep.
            cache: Optional ResponseCache instance.
            metrics: Optional MetricsCollector instance.
            prompt_optimizer: Optional PromptOptimizer instance.
        """
        self.retriever = retriever
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name
        self.max_history = max_history
        self.conversation_history: List[Tuple[str, str]] = []

        # Enhanced components
        self.cache = cache
        self.metrics = metrics
        self.prompt_optimizer = prompt_optimizer or PromptOptimizer(
            openai_client=self.openai_client,
            auto_augment_short_queries=False,  # Disable query augmentation to avoid semantic mismatch
            validate_output=False,  # Disable guardrails validation - trust prompt engineering
        )

    def _format_context_with_quality_tiers(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents as a simple numbered list.

        Trust the retrieval system - all retrieved sources are valid and should be used.
        This method replaces the previous threshold-based filtering approach.

        Args:
            retrieved_docs: List of retrieved document chunks.

        Returns:
            Formatted context string with all sources.
        """
        if not retrieved_docs:
            return "No relevant information was found in the database."

        context_parts = []
        context_parts.append("## RETRIEVED SOURCES\n")

        for idx, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "")
            url = doc.get("url", "")
            title = doc.get("title", "Sin título")

            context_parts.append(
                f"[Source {idx}]\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}\n"
            )

        context_parts.append("\n## INSTRUCTIONS:")
        context_parts.append("- Use the sources above to answer the question")
        context_parts.append("- Cite sources using [N] format")
        context_parts.append(
            "- Prioritize the most relevant snippets; skip weakly-related sources to avoid citation noise")
        context_parts.append(
            "- If the sources don't contain the specific information asked, acknowledge what's missing")
        context_parts.append("- Be direct and factual based on what's in the sources")

        return "\n".join(context_parts)

    def _normalize_question(self, question: str) -> str:
        """
        Normalize user question to reduce sensitivity to trailing punctuation/spacing.
        """
        cleaned = question.strip()
        cleaned = re.sub(r"[\s?¿!¡]+$", "", cleaned)
        cleaned = " ".join(cleaned.split())
        return cleaned

    def _strip_sources_section(self, response: str) -> str:
        """
        Remove any existing sources footer to avoid duplicate or mismatched sections.
        """
        markers = [r"\*\*Sources consulted:\*\*", r"\*\*Sources:\*\*"]
        pattern = re.compile("|".join(markers), re.IGNORECASE)
        match = pattern.search(response)
        if match:
            return response[: match.start()].rstrip()
        return response.rstrip()

    def _normalize_response_citations(
            self, response: str, retrieved_docs: List[Dict]
    ) -> str:
        """
        Remove any sources footer. The UI shows sources separately.
        """
        _ = retrieved_docs  # kept for signature compatibility
        return self._strip_sources_section(response)

    def _format_chat_history(self) -> str:
        """
        Format conversation history for the prompt.

        Returns:
            Formatted chat history string.
        """
        if not self.conversation_history:
            return "No prior conversation."

        history_parts = []
        for question, answer in self.conversation_history[-self.max_history:]:
            history_parts.append(f"User: {question}\nAssistant: {answer}")

        return "\n\n".join(history_parts)

    def _generate_response(self, messages: List[Dict[str, str]]) -> Tuple[str, Optional[Dict]]:
        """
        Generate response using OpenAI API.

        Args:
            messages: Chat messages containing system and user instructions.

        Returns:
            Tuple of (generated response, usage stats).
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )

            content = response.choices[0].message.content.strip()

            # Extract usage stats if available
            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return content, usage

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}", None

    def query(
            self, question: str, top_k: int = None
    ) -> Tuple[str, List[Dict]]:
        """
        Query the RAG system with a question.
        Enhanced with caching, metrics, and prompt optimization.

        Args:
            question: User question.
            top_k: Number of documents to retrieve (optional override).

        Returns:
            Tuple of (response, source_documents).
        """
        k = top_k or 5
        normalized_question = self._normalize_question(question)

        # Start metrics trace if available
        trace = None
        if self.metrics:
            trace = self.metrics.trace_pipeline(
                normalized_question,
                metadata={"top_k": k, "model": self.model_name}
            )
            trace_ctx = trace.__enter__()

        try:
            logger.info(f"Processing query: {question[:50]}...")

            if self.cache:
                cached = self.cache.get(normalized_question, k)
                if cached:
                    logger.info("Returning cached response")
                    if self.metrics:
                        self.metrics.record_cache_hit()
                    return cached["response"], cached["sources"]
                else:
                    if self.metrics:
                        self.metrics.record_cache_miss()

            if trace:
                with self.metrics.measure_stage(trace_ctx, "retrieval"):
                    retrieved_docs = self.retriever.retrieve(normalized_question, top_k=k)
            else:
                retrieved_docs = self.retriever.retrieve(normalized_question, top_k=k)

            # Log retrieval results for debugging
            scores = [d.get("relevance_score", d.get("similarity", 0)) for d in retrieved_docs]
            logger.info(
                f"Retrieved {len(retrieved_docs)} documents. "
                f"Scores: {[f'{score:.2f}' for score in scores]}"
            )
            if not retrieved_docs:
                logger.warning(f"No documents retrieved for query: '{question}'")

            if self.metrics:
                self.metrics.record_retrieval(len(retrieved_docs), scores)

            # Handle empty retrieval explicitly with informative message
            if not retrieved_docs:
                logger.error(f"Empty retrieval for query: '{question}'")
                return (
                    "I couldn't find relevant CV information for your query. "
                    "This might mean: (1) the query doesn't match any CV content, "
                    "(2) the database is empty, or (3) there's a retrieval issue. "
                    "Please try rephrasing your question or check that CVs have been loaded.",
                    []
                )

            context = self._format_context_with_quality_tiers(retrieved_docs)
            chat_history = self._format_chat_history()

            if trace:
                with self.metrics.measure_stage(trace_ctx, "prompt_creation"):
                    messages, prompt_metadata = self.prompt_optimizer.create_prompt(
                        context=context,
                        chat_history=chat_history,
                        question=question,
                        chat_history_list=self.conversation_history,
                    )
            else:
                messages, prompt_metadata = self.prompt_optimizer.create_prompt(
                    context=context,
                    chat_history=chat_history,
                    question=question,
                    chat_history_list=self.conversation_history,
                )

            # Log prompt analysis
            logger.info(
                f"Prompt analysis: lang={prompt_metadata['language']}, "
                f"complexity={prompt_metadata['complexity']}, "
                f"augmented={prompt_metadata['was_augmented']}"
            )

            # Generate response
            if trace:
                with self.metrics.measure_stage(trace_ctx, "generation"):
                    response, usage = self._generate_response(messages)
            else:
                response, usage = self._generate_response(messages)

            # Record token usage
            if self.metrics and usage:
                self.metrics.record_token_usage(
                    usage["prompt_tokens"],
                    usage["completion_tokens"],
                )

            # Validate response with detected language
            if trace:
                with self.metrics.measure_stage(trace_ctx, "validation"):
                    validation = self.prompt_optimizer.validate_response(
                        response, context, retrieved_docs,
                        language=prompt_metadata['language']
                    )
            else:
                validation = self.prompt_optimizer.validate_response(
                    response, context, retrieved_docs,
                    language=prompt_metadata['language']
                )

            if not validation["passed"]:
                logger.warning(
                    f"Response validation issues: {validation['issues']}"
                )

            normalized_response = self._normalize_response_citations(
                response, retrieved_docs
            )

            # Update conversation history
            self.conversation_history.append((question, normalized_response))

            # Trim history if needed
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]

            # Cache response
            if self.cache:
                self.cache.set(normalized_question, k, normalized_response, retrieved_docs)

            logger.info(
                f"Generated response with {len(retrieved_docs)} sources "
                f"(validation score: {validation.get('score', 0):.2f})"
            )

            return normalized_response, retrieved_docs

        finally:
            if trace:
                trace.__exit__(None, None, None)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_history_length(self) -> int:
        """
        Get current conversation history length.

        Returns:
            Number of conversation turns.
        """
        return len(self.conversation_history)
