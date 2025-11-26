"""
Embedding generation module using OpenAI API.

This module handles generating embeddings for text chunks with
batching and retry logic.
"""

import logging
import time
from typing import List, Dict
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using OpenAI's embedding models.
    """

    def __init__(
            self,
            api_key: str,
            model: str = "text-embedding-3-small",
            batch_size: int = 100,
    ) -> None:
        """
        Initialize the embedding generator.

        Args:
            api_key: OpenAI API key.
            model: Name of the embedding model to use.
            batch_size: Number of texts to embed in each batch.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

        logger.info(f"Initialized embedding generator with model={model}")

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
    )
    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with retry logic.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            Exception: If embedding generation fails after retries.
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model,
            )

            embeddings = [item.embedding for item in response.data]
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of chunk dictionaries with 'chunk_text' key.

        Returns:
            List of chunks with added 'embedding' key.
        """
        if not chunks:
            logger.warning("No chunks to embed")
            return []

        total_chunks = len(chunks)
        logger.info(f"Generating embeddings for {total_chunks} chunks")

        for i in range(0, total_chunks, self.batch_size):
            batch = chunks[i: i + self.batch_size]
            batch_texts = [chunk["chunk_text"] for chunk in batch]

            try:
                logger.info(
                    f"Processing batch {i // self.batch_size + 1} "
                    f"({i + 1}-{min(i + len(batch), total_chunks)} of {total_chunks})"
                )

                embeddings = self._generate_embeddings_batch(batch_texts)

                for chunk, embedding in zip(batch, embeddings):
                    chunk["embedding"] = embedding

                if i + self.batch_size < total_chunks:
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed to embed batch starting at index {i}: {e}")

                for chunk in batch:
                    chunk["embedding"] = None
                    chunk["error"] = str(e)

        successful = sum(1 for chunk in chunks if chunk.get("embedding") is not None)
        logger.info(
            f"Successfully generated {successful}/{total_chunks} embeddings"
        )

        return chunks

    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        embeddings = self._generate_embeddings_batch([text])
        return embeddings[0]
