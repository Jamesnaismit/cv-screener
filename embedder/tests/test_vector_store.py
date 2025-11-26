"""
Unit tests for the vector store.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from database.vector_store import VectorStore


def create_mock_db():
    """Helper to create mock database connection."""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
    mock_conn.closed = False
    return mock_conn, mock_cursor


@pytest.mark.unit
class TestVectorStore:
    """Test cases for VectorStore."""

    @patch('database.vector_store.psycopg2.connect')
    def test_vector_store_initialization(self, mock_connect):
        """Test vector store initialization."""
        mock_connect.return_value = Mock()

        store = VectorStore(database_url="postgresql://test", ensure_schema=False)

        assert store.database_url == "postgresql://test"
        mock_connect.assert_called_once_with("postgresql://test")

    @patch('database.vector_store.psycopg2.connect')
    def test_document_exists_true(self, mock_connect):
        """Test checking if document exists (returns True)."""
        mock_conn, mock_cursor = create_mock_db()
        mock_connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = (True,)

        store = VectorStore(database_url="postgresql://test", ensure_schema=False)
        exists = store.document_exists("https://test.com")

        assert exists is True
        # one call for select exists
        assert mock_cursor.execute.call_count == 1

    @patch('database.vector_store.psycopg2.connect')
    def test_document_exists_false(self, mock_connect):
        """Test checking if document exists (returns False)."""
        mock_conn, mock_cursor = create_mock_db()
        mock_connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = (False,)

        store = VectorStore(database_url="postgresql://test", ensure_schema=False)
        exists = store.document_exists("https://nonexistent.com")

        assert exists is False

    @patch('database.vector_store.psycopg2.connect')
    def test_get_document_id_found(self, mock_connect):
        """Test getting document ID when it exists."""
        mock_conn, mock_cursor = create_mock_db()
        mock_connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = (123,)

        store = VectorStore(database_url="postgresql://test", ensure_schema=False)
        doc_id = store.get_document_id("https://test.com")

        assert doc_id == 123

    @patch('database.vector_store.psycopg2.connect')
    def test_get_document_id_not_found(self, mock_connect):
        """Test getting document ID when it doesn't exist."""
        mock_conn, mock_cursor = create_mock_db()
        mock_connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = None

        store = VectorStore(database_url="postgresql://test", ensure_schema=False)
        doc_id = store.get_document_id("https://nonexistent.com")

        assert doc_id is None

    @patch('database.vector_store.psycopg2.connect')
    def test_insert_document(self, mock_connect):
        """Test inserting a new document."""
        mock_conn, mock_cursor = create_mock_db()
        mock_connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = (456,)

        store = VectorStore(database_url="postgresql://test", ensure_schema=False)

        doc_id = store.insert_document(
            url="https://test.com",
            title="Test Title",
            content="Test content",
            metadata={"language": "en"},
            content_hash="abc123"
        )

        assert doc_id == 456
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch('database.vector_store.psycopg2.connect')
    def test_delete_embeddings_for_document(self, mock_connect):
        """Test deleting embeddings for a document."""
        mock_conn, mock_cursor = create_mock_db()
        mock_connect.return_value = mock_conn

        store = VectorStore(database_url="postgresql://test", ensure_schema=False)
        store.delete_embeddings_for_document(123)

        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch('database.vector_store.psycopg2.connect')
    def test_insert_embeddings_empty(self, mock_connect):
        """Test inserting empty embeddings list."""
        mock_conn, mock_cursor = create_mock_db()
        mock_connect.return_value = mock_conn

        store = VectorStore(database_url="postgresql://test", ensure_schema=False)
        count = store.insert_embeddings([])

        assert count == 0

    @patch('database.vector_store.psycopg2.connect')
    def test_insert_embeddings_without_embedding_field(self, mock_connect):
        """Test inserting chunks without embeddings."""
        mock_conn, mock_cursor = create_mock_db()
        mock_connect.return_value = mock_conn

        store = VectorStore(database_url="postgresql://test", ensure_schema=False)

        chunks = [
            {"chunk_text": "test", "chunk_index": 0, "url": "test.com", "title": "Test"}
        ]

        count = store.insert_embeddings(chunks)

        assert count == 0

    @patch('database.vector_store.psycopg2.connect')
    def test_insert_embeddings_success(self, mock_connect):
        """Test successful embedding insertion."""
        mock_conn, mock_cursor = create_mock_db()
        mock_connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = (1,)  # document_id

        store = VectorStore(database_url="postgresql://test", ensure_schema=False)

        chunks = [
            {
                "chunk_text": "Test chunk",
                "chunk_index": 0,
                "url": "https://test.com",
                "title": "Test",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {"language": "en"}
            }
        ]

        count = store.insert_embeddings(chunks)

        assert count == 1
        assert mock_cursor.execute.call_count >= 1

    @patch('database.vector_store.psycopg2.connect')
    def test_get_document_count(self, mock_connect):
        """Test getting document count."""
        mock_conn, mock_cursor = create_mock_db()
        mock_connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = (42,)

        store = VectorStore(database_url="postgresql://test", ensure_schema=False)
        count = store.get_document_count()

        assert count == 42

    @patch('database.vector_store.psycopg2.connect')
    def test_get_embedding_count(self, mock_connect):
        """Test getting embedding count."""
        mock_conn, mock_cursor = create_mock_db()
        mock_connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = (100,)

        store = VectorStore(database_url="postgresql://test", ensure_schema=False)
        count = store.get_embedding_count()

        assert count == 100

    @patch('database.vector_store.psycopg2.connect')
    def test_search_similar(self, mock_connect):
        """Test similarity search."""
        mock_conn, mock_cursor = create_mock_db()
        mock_connect.return_value = mock_conn
        mock_cursor.fetchall.return_value = [
            ("Chunk 1", {"index": 0}, "https://test1.com", "Test 1", 0.95),
            ("Chunk 2", {"index": 1}, "https://test2.com", "Test 2", 0.87)
        ]

        store = VectorStore(database_url="postgresql://test", ensure_schema=False)

        query_embedding = [0.1] * 1536
        results = store.search_similar(query_embedding, top_k=2)

        assert len(results) == 2
        assert results[0]["chunk_text"] == "Chunk 1"
        assert results[0]["url"] == "https://test1.com"
        assert results[0]["similarity"] == 0.95
        assert results[1]["chunk_text"] == "Chunk 2"
        assert results[1]["similarity"] == 0.87

    @patch('database.vector_store.psycopg2.connect')
    def test_ensure_connection_reconnects(self, mock_connect):
        """Test that ensure_connection reconnects when closed."""
        mock_conn = Mock()
        mock_conn.closed = True
        mock_connect.return_value = mock_conn

        store = VectorStore(database_url="postgresql://test", ensure_schema=False)
        store.conn.closed = True

        store._ensure_connection()

        assert mock_connect.call_count == 2

    @patch('database.vector_store.psycopg2.connect')
    def test_close_connection(self, mock_connect):
        """Test closing database connection."""
        mock_conn = Mock()
        mock_conn.closed = False
        mock_connect.return_value = mock_conn

        store = VectorStore(database_url="postgresql://test")
        store.close()

        mock_conn.close.assert_called_once()
