"""
Vector store module for PostgreSQL with pgvector.

This module handles storing and retrieving document embeddings
in PostgreSQL using the pgvector extension.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values, Json

logger = logging.getLogger(__name__)


class VectorStore:
    """
    PostgreSQL vector store for document embeddings.
    """

    def __init__(self, database_url: str, ensure_schema: bool = False) -> None:
        """
        Initialize the vector store.

        Args:
            database_url: PostgreSQL connection URL.
            ensure_schema: Whether to create/ensure schema on init.
        """
        self.database_url = database_url
        self.conn = None
        self.ensure_schema = ensure_schema
        self._connect()

    def _connect(self) -> None:
        """Establish database connection."""
        created_db = False
        try:
            self.conn = psycopg2.connect(self.database_url)
            logger.info("Successfully connected to database")
        except Exception as e:
            if "does not exist" in str(e).lower():
                logger.warning(f"Database missing, creating it: {e}")
                self._create_database_if_missing()
                created_db = True
                self.conn = psycopg2.connect(self.database_url)
                logger.info("Successfully connected after creating database")
            else:
                logger.error(f"Failed to connect to database: {e}")
                raise
        if created_db or self.ensure_schema:
            self._ensure_schema()

    def _create_database_if_missing(self) -> None:
        """Create the target database if it does not exist."""
        parsed = urlparse(self.database_url)
        db_name = (parsed.path or "").lstrip("/")
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432
        user = parsed.username or ""
        password = parsed.password or ""

        admin_dsn = f"dbname=postgres user={user} password={password} host={host} port={port}"

        conn = psycopg2.connect(admin_dsn)
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (db_name,),
                )
                exists = cur.fetchone()
                if exists:
                    logger.info("Database already exists, skipping creation")
                    return

                cur.execute(
                    sql.SQL('CREATE DATABASE {} OWNER {}').format(
                        sql.Identifier(db_name),
                        sql.Identifier(user),
                    )
                )
                logger.info(f"Database '{db_name}' created")
        finally:
            conn.close()

    def _ensure_connection(self) -> None:
        """Ensure database connection is active."""
        if self.conn is None or self.conn.closed:
            logger.warning("Database connection lost, reconnecting...")
            self._connect()

    def _ensure_schema(self) -> None:
        """Ensure required tables exist (documents, embeddings)."""
        self._ensure_connection()
        cur = self.conn.cursor()
        try:
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    content_hash TEXT,
                    last_ingested_at TIMESTAMP DEFAULT NOW(),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding vector(1536) NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
                CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
                CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON embeddings(document_id);
                CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin(metadata);
                CREATE INDEX IF NOT EXISTS idx_embeddings_metadata ON embeddings USING gin(metadata);
            """)
            self.conn.commit()
        finally:
            cur.close()
        logger.info("Schema ensured (documents, embeddings).")

    def close(self) -> None:
        """Close database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Database connection closed")

    def document_exists(self, url: str) -> bool:
        """
        Check if a document with the given URL already exists.

        Args:
            url: Document URL to check.

        Returns:
            True if document exists, False otherwise.
        """
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM documents WHERE url = %s)", (url,))
            return cur.fetchone()[0]

    def get_document_id(self, url: str) -> Optional[int]:
        """
        Get document ID by URL.

        Args:
            url: Document URL.

        Returns:
            Document ID if found, None otherwise.
        """
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM documents WHERE url = %s", (url,))
            result = cur.fetchone()
            return result[0] if result else None

    def get_document_info(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata (id, content_hash) by URL.

        Args:
            url: Document URL.

        Returns:
            Dict with id and content_hash or None.
        """
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_hash FROM documents WHERE url = %s",
                (url,),
            )
            result = cur.fetchone()
            if result:
                return {"id": result[0], "content_hash": result[1]}
        return None

    def insert_document(
            self, url: str, title: str, content: str, metadata: Dict, content_hash: str
    ) -> int:
        """
        Insert a new document into the database.

        Args:
            url: Document URL.
            title: Document title.
            content: Document content.
            metadata: Document metadata.
            content_hash: Hash of content for change detection.

        Returns:
            ID of the inserted document.
        """
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (url, title, content, metadata, content_hash)
                VALUES (%s, %s, %s, %s, %s) ON CONFLICT (url) DO
                UPDATE
                    SET title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    content_hash = EXCLUDED.content_hash,
                    last_ingested_at = NOW(),
                    updated_at = NOW()
                    RETURNING id
                """,
                (url, title, content, Json(metadata), content_hash),
            )
            document_id = cur.fetchone()[0]
            self.conn.commit()

        return document_id

    def delete_embeddings_for_document(self, document_id: int) -> None:
        """
        Delete all embeddings for a document.

        Args:
            document_id: ID of the document.
        """
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM embeddings WHERE document_id = %s",
                (document_id,)
            )
            self.conn.commit()

        logger.debug(f"Deleted embeddings for document {document_id}")

    def insert_embeddings(self, chunks: List[Dict]) -> int:
        """
        Insert embeddings for document chunks.

        Args:
            chunks: List of chunk dictionaries with embeddings.

        Returns:
            Number of embeddings inserted.
        """
        self._ensure_connection()

        if not chunks:
            return 0

        valid_chunks = [c for c in chunks if c.get("embedding") is not None]

        if not valid_chunks:
            logger.warning("No valid embeddings to insert")
            return 0

        inserted_count = 0

        for chunk in valid_chunks:
            try:

                document_id = self.get_document_id(chunk["url"])

                if document_id is None:
                    document_id = self.insert_document(
                        url=chunk["url"],
                        title=chunk["title"],
                        content=chunk["chunk_text"],
                        metadata=chunk.get("metadata", {}),
                        content_hash="",
                    )

                with self.conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO embeddings
                            (document_id, chunk_index, chunk_text, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            document_id,
                            chunk["chunk_index"],
                            chunk["chunk_text"],
                            chunk["embedding"],
                            Json(chunk.get("metadata", {})),
                        ),
                    )
                    self.conn.commit()
                    inserted_count += 1

            except Exception as e:
                logger.error(f"Error inserting embedding for chunk: {e}")
                self.conn.rollback()

        logger.info(f"Inserted {inserted_count} embeddings")
        return inserted_count

    def get_document_count(self) -> int:
        """
        Get total number of documents.

        Returns:
            Document count.
        """
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            return cur.fetchone()[0]

    def get_embedding_count(self) -> int:
        """
        Get total number of embeddings.

        Returns:
            Embedding count.
        """
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM embeddings")
            return cur.fetchone()[0]

    def search_similar(
            self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of similar document chunks with metadata.
        """
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.chunk_text,
                       e.metadata,
                       d.url,
                       d.title,
                       1 - (e.embedding <=> %s::vector) as similarity
                FROM embeddings e
                         JOIN documents d ON e.document_id = d.id
                ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )

            results = []
            for row in cur.fetchall():
                results.append({
                    "chunk_text": row[0],
                    "metadata": row[1],
                    "url": row[2],
                    "title": row[3],
                    "similarity": float(row[4]),
                })

            return results
