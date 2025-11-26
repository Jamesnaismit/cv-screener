"""
Unit tests for the vector retriever.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from rag.retriever import VectorRetriever


def create_mock_db():
    """Helper to create mock database connection."""
    conn = Mock()
    cursor = Mock()
    conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
    conn.cursor.return_value.__exit__ = Mock(return_value=False)
    conn.closed = False
    return conn, cursor


@pytest.mark.unit
class TestVectorRetriever:
    """Test cases for VectorRetriever."""

    @patch('rag.retriever.psycopg2.connect')
    @patch('rag.retriever.OpenAI')
    def test_retriever_initialization(self, mock_openai, mock_connect):
        """Test retriever initialization."""
        mock_connect.return_value = Mock()

        retriever = VectorRetriever(
            database_url="postgresql://test",
            openai_api_key="test_key",
            embedding_model="text-embedding-3-small",
            top_k=5
        )

        assert retriever.database_url == "postgresql://test"
        assert retriever.top_k == 5
        assert retriever.embedding_model == "text-embedding-3-small"
        mock_openai.assert_called_once_with(api_key="test_key")
        mock_connect.assert_called_once()

    @patch('rag.retriever.psycopg2.connect')
    @patch('rag.retriever.OpenAI')
    def test_generate_query_embedding(self, mock_openai, mock_connect):
        """Test query embedding generation."""
        mock_connect.return_value = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        retriever = VectorRetriever(
            database_url="postgresql://test",
            openai_api_key="test_key"
        )

        embedding = retriever.generate_query_embedding("test query")

        assert embedding == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            input="test query",
            model="text-embedding-3-small"
        )

    @patch('rag.retriever.psycopg2.connect')
    @patch('rag.retriever.OpenAI')
    def test_retrieve_with_results(self, mock_openai, mock_connect):
        """Test retrieval with results."""
        mock_conn, mock_cursor = create_mock_db()
        mock_connect.return_value = mock_conn

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        mock_cursor.fetchall.return_value = [
            (
                "Test content 1",
                {"chunk_index": 0},
                "https://test1.com",
                "Test Title 1",
                0.95
            ),
            (
                "Test content 2",
                {"chunk_index": 1},
                "https://test2.com",
                "Test Title 2",
                0.87
            )
        ]

        retriever = VectorRetriever(
            database_url="postgresql://test",
            openai_api_key="test_key",
            top_k=2
        )

        results = retriever.retrieve("test query")

        assert len(results) == 2
        assert results[0]["content"] == "Test content 1"
        assert results[0]["url"] == "https://test1.com"
        assert results[0]["title"] == "Test Title 1"
        assert results[0]["similarity"] == 0.95
        assert results[1]["content"] == "Test content 2"
        assert results[1]["similarity"] == 0.87

    @patch('rag.retriever.psycopg2.connect')
    @patch('rag.retriever.OpenAI')
    def test_retrieve_custom_top_k(self, mock_openai, mock_connect):
        """Test retrieval with custom top_k."""
        mock_conn, mock_cursor = create_mock_db()
        mock_connect.return_value = mock_conn

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        mock_cursor.fetchall.return_value = []

        retriever = VectorRetriever(
            database_url="postgresql://test",
            openai_api_key="test_key",
            top_k=5
        )

        retriever.retrieve("test query", top_k=10)

        call_args = mock_cursor.execute.call_args[0]
        assert call_args[1][2] == 10

    @patch('rag.retriever.psycopg2.connect')
    @patch('rag.retriever.OpenAI')
    def test_retrieve_no_results(self, mock_openai, mock_connect):
        """Test retrieval with no results."""
        mock_conn, mock_cursor = create_mock_db()
        mock_connect.return_value = mock_conn

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        mock_cursor.fetchall.return_value = []

        retriever = VectorRetriever(
            database_url="postgresql://test",
            openai_api_key="test_key"
        )

        results = retriever.retrieve("test query")

        assert len(results) == 0

    @patch('rag.retriever.psycopg2.connect')
    @patch('rag.retriever.OpenAI')
    def test_ensure_connection_reconnects(self, mock_openai, mock_connect):
        """Test that ensure_connection reconnects when connection is closed."""
        mock_conn = Mock()
        mock_conn.closed = True
        mock_connect.return_value = mock_conn

        mock_openai.return_value = Mock()

        retriever = VectorRetriever(
            database_url="postgresql://test",
            openai_api_key="test_key"
        )

        retriever.conn.closed = True

        retriever._ensure_connection()

        assert mock_connect.call_count == 2

    @patch('rag.retriever.psycopg2.connect')
    @patch('rag.retriever.OpenAI')
    def test_close_connection(self, mock_openai, mock_connect):
        """Test closing database connection."""
        mock_conn = Mock()
        mock_conn.closed = False
        mock_connect.return_value = mock_conn

        mock_openai.return_value = Mock()

        retriever = VectorRetriever(
            database_url="postgresql://test",
            openai_api_key="test_key"
        )

        retriever.close()

        mock_conn.close.assert_called_once()
