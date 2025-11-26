"""
Vector retriever module for semantic search.

This module handles retrieving relevant document chunks from the
vector database based on query similarity.
"""

import logging
from typing import List, Dict
import psycopg2
from openai import OpenAI

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Retrieves relevant document chunks using vector similarity search.
    """

    def __init__(
        self,
        database_url: str,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 5,
    ) -> None:
        """
        Initialize the vector retriever.

        Args:
            database_url: PostgreSQL connection URL.
            openai_api_key: OpenAI API key for generating query embeddings.
            embedding_model: Name of the embedding model.
            top_k: Number of results to retrieve.
        """
        self.database_url = database_url
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.conn = None
        self._connect()

    def _connect(self) -> None:
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(self.database_url)
            logger.info("Retriever connected to database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _ensure_connection(self) -> None:
        """Ensure database connection is active."""
        if self.conn is None or self.conn.closed:
            logger.warning("Database connection lost, reconnecting...")
            self._connect()

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            query: Query text.

        Returns:
            Query embedding vector.
        """
        try:
            response = self.openai_client.embeddings.create(
                input=query,
                model=self.embedding_model,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: Query text.
            top_k: Number of results to retrieve (overrides default if provided).

        Returns:
            List of relevant document chunks with metadata and similarity scores.
        """
        k = top_k if top_k is not None else self.top_k

        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)

        # Search for similar chunks
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    e.chunk_text,
                    e.metadata,
                    d.url,
                    d.title,
                    1 - (e.embedding <=> %s::vector) as similarity
                FROM embeddings e
                JOIN documents d ON e.document_id = d.id
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, k),
            )

            results = []
            for row in cur.fetchall():
                results.append({
                    "content": row[0],
                    "metadata": row[1],
                    "url": row[2],
                    "title": row[3],
                    "similarity": float(row[4]),
                })

        logger.info(f"Retrieved {len(results)} results for query: {query[:50]}...")
        return results

    def close(self) -> None:
        """Close database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Retriever database connection closed")
