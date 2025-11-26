"""
Conversational RAG chain module.

This module implements the RAG pipeline with conversation memory,
combining retrieval with LLM generation for context-aware responses.
Enhanced with caching, metrics, re-ranking, and prompt optimization.
"""

import logging
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

from .cache import ResponseCache
from .metrics import MetricsCollector
from .prompts import PromptOptimizer

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

    # SYSTEM_PROMPT removed - now handled by PromptOptimizer in prompts.py

    def __init__(
        self,
        retriever,
        openai_api_key: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000,
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
            temperature: Temperature for response generation.
            max_tokens: Maximum tokens in response.
            max_history: Maximum number of conversation turns to keep.
            cache: Optional ResponseCache instance.
            metrics: Optional MetricsCollector instance.
            prompt_optimizer: Optional PromptOptimizer instance.
        """
        self.retriever = retriever
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_history = max_history
        self.conversation_history: List[Tuple[str, str]] = []
        
        # Enhanced components
        self.cache = cache
        self.metrics = metrics
        self.prompt_optimizer = prompt_optimizer or PromptOptimizer(
            openai_client=self.openai_client
        )

    def _format_context_with_quality_tiers(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents with quality tiers for better semantic grounding.

        Separates sources into HIGH/MEDIUM/LOW relevance tiers with explicit
        instructions on how to use each tier.

        Args:
            retrieved_docs: List of retrieved document chunks.

        Returns:
            Formatted context string with quality tiers.
        """
        if not retrieved_docs:
            return "No relevant information was found in the database."

        # Separate documents by relevance threshold
        high_relevance = [d for d in retrieved_docs if d.get("similarity", 0) >= 0.75]
        medium_relevance = [d for d in retrieved_docs if 0.5 <= d.get("similarity", 0) < 0.75]
        low_relevance = [d for d in retrieved_docs if d.get("similarity", 0) < 0.5]

        context_parts = []
        source_index = 1

        if high_relevance:
            context_parts.append("## HIGH RELEVANCE SOURCES (Primary References)")
            context_parts.append("Use these sources for direct claims:\n")

            for doc in high_relevance:
                similarity = doc.get("similarity", 0)
                content = doc.get("content", "")
                url = doc.get("url", "")
                title = doc.get("title", "Sin título")

                context_parts.append(
                    f"[Source {source_index}] ({similarity:.1%} match)\n"
                    f"Title: {title}\n"
                    f"URL: {url}\n"
                    f"Content: {content}\n"
                )
                source_index += 1

        if medium_relevance:
            context_parts.append("\n## MEDIUM RELEVANCE SOURCES (Supportive Context)")
            context_parts.append("Use carefully and indicate medium confidence:\n")

            for doc in medium_relevance:
                similarity = doc.get("similarity", 0)
                content = doc.get("content", "")
                url = doc.get("url", "")
                title = doc.get("title", "Sin título")

                context_parts.append(
                    f"[Source {source_index}] ({similarity:.1%} match - use with caution)\n"
                    f"Title: {title}\n"
                    f"URL: {url}\n"
                    f"Content: {content}\n"
                )
                source_index += 1

        if low_relevance and len(high_relevance) + len(medium_relevance) < 3:
            context_parts.append("\n## LOW RELEVANCE SOURCES (Background Only)")
            context_parts.append("DO NOT CITE DIRECTLY. Background use only:\n")

            for doc in low_relevance:
                similarity = doc.get("similarity", 0)
                content = doc.get("content", "")
                title = doc.get("title", "Sin título")

                truncated_content = content[:300] + "..." if len(content) > 300 else content

                context_parts.append(
                    f"[Source {source_index}] ({similarity:.1%} - background only, do not cite)\n"
                    f"Title: {title}\n"
                    f"Content: {truncated_content}\n"
                )
                source_index += 1

        context_parts.append("\n## SOURCE USAGE RULES:")
        context_parts.append("- Prioritize HIGH RELEVANCE for direct statements")
        context_parts.append("- Use MEDIUM RELEVANCE for supporting details; mark medium confidence")
        context_parts.append("- Use LOW RELEVANCE only for background, never for specific facts")
        context_parts.append("- If all sources have relevance < 35%, omit the 'Sources consulted' section and acknowledge lack of info")
        context_parts.append("- Be clear about gaps and avoid speculation")

        return "\n".join(context_parts)

    def _format_chat_history(self) -> str:
        """
        Format conversation history for the prompt.

        Returns:
            Formatted chat history string.
        """
        if not self.conversation_history:
            return "No prior conversation."

        history_parts = []
        for question, answer in self.conversation_history[-self.max_history :]:
            history_parts.append(f"User: {question}\nAssistant: {answer}")

        return "\n\n".join(history_parts)

    def _generate_response(self, prompt: str) -> Tuple[str, Optional[Dict]]:
        """
        Generate response using OpenAI API.

        Args:
            prompt: Complete prompt with context and question.

        Returns:
            Tuple of (generated response, usage stats).
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
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
        
        # Start metrics trace if available
        trace = None
        if self.metrics:
            trace = self.metrics.trace_pipeline(
                question,
                metadata={"top_k": k, "model": self.model_name}
            )
            trace_ctx = trace.__enter__()
        
        try:
            logger.info(f"Processing query: {question[:50]}...")

            if self.cache:
                cached = self.cache.get(question, k)
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
                    retrieved_docs = self.retriever.retrieve(question, top_k=k)
            else:
                retrieved_docs = self.retriever.retrieve(question, top_k=k)

            if self.metrics:
                similarities = [d.get("similarity", 0) for d in retrieved_docs]
                self.metrics.record_retrieval(len(retrieved_docs), similarities)

            context = self._format_context_with_quality_tiers(retrieved_docs)
            chat_history = self._format_chat_history()

            if trace:
                with self.metrics.measure_stage(trace_ctx, "prompt_creation"):
                    prompt, prompt_metadata = self.prompt_optimizer.create_prompt(
                        context=context,
                        chat_history=chat_history,
                        question=question,
                        chat_history_list=self.conversation_history,
                    )
            else:
                prompt, prompt_metadata = self.prompt_optimizer.create_prompt(
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
                    response, usage = self._generate_response(prompt)
            else:
                response, usage = self._generate_response(prompt)

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

            # Update conversation history
            self.conversation_history.append((question, response))

            # Trim history if needed
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]

            # Cache response
            if self.cache:
                self.cache.set(question, k, response, retrieved_docs)

            logger.info(
                f"Generated response with {len(retrieved_docs)} sources "
                f"(validation score: {validation.get('score', 0):.2f})"
            )

            return response, retrieved_docs

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
