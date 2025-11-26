"""
Unit tests for the embedding generator.
"""

import pytest
from unittest.mock import Mock, patch
from processors.embedding_generator import EmbeddingGenerator


@pytest.fixture
def mock_openai_client():
    """Create mock OpenAI client."""
    client = Mock()
    return client


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "chunk_text": "Test chunk 1",
            "chunk_index": 0,
            "url": "cv://sample-candidate",
            "title": "Test",
            "metadata": {}
        },
        {
            "chunk_text": "Test chunk 2",
            "chunk_index": 1,
            "url": "cv://sample-candidate",
            "title": "Test",
            "metadata": {}
        }
    ]


@pytest.mark.unit
class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator."""

    @patch('processors.embedding_generator.OpenAI')
    def test_generator_initialization(self, mock_openai):
        """Test generator initialization."""
        generator = EmbeddingGenerator(
            api_key="test_key",
            model="text-embedding-3-small",
            batch_size=100
        )

        assert generator.model == "text-embedding-3-small"
        assert generator.batch_size == 100
        mock_openai.assert_called_once_with(api_key="test_key")

    @patch('processors.embedding_generator.OpenAI')
    def test_generate_embeddings_batch(self, mock_openai):
        """Test batch embedding generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator(api_key="test_key")

        texts = ["text 1", "text 2"]
        embeddings = generator._generate_embeddings_batch(texts)

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

        mock_client.embeddings.create.assert_called_once_with(
            input=texts,
            model="text-embedding-3-small"
        )

    @patch('processors.embedding_generator.OpenAI')
    @patch('processors.embedding_generator.time.sleep')
    def test_generate_embeddings_success(self, mock_sleep, mock_openai, sample_chunks):
        """Test successful embedding generation for chunks."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator(api_key="test_key")

        result_chunks = generator.generate_embeddings(sample_chunks)

        assert len(result_chunks) == 2
        assert result_chunks[0]["embedding"] == [0.1, 0.2, 0.3]
        assert result_chunks[1]["embedding"] == [0.4, 0.5, 0.6]

    @patch('processors.embedding_generator.OpenAI')
    def test_generate_embeddings_empty_list(self, mock_openai):
        """Test handling of empty chunk list."""
        mock_openai.return_value = Mock()

        generator = EmbeddingGenerator(api_key="test_key")
        result = generator.generate_embeddings([])

        assert result == []

    @patch('processors.embedding_generator.OpenAI')
    @patch('processors.embedding_generator.time.sleep')
    def test_generate_embeddings_batch_error(self, mock_sleep, mock_openai):
        """Test handling of batch errors."""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator(api_key="test_key")

        chunks = [
            {"chunk_text": "test", "chunk_index": 0, "url": "cv://sample-candidate", "title": "Test", "metadata": {}}
        ]

        result = generator.generate_embeddings(chunks)

        assert result[0].get("embedding") is None
        assert "error" in result[0]

    @patch('processors.embedding_generator.OpenAI')
    @patch('processors.embedding_generator.time.sleep')
    def test_generate_embeddings_batching(self, mock_sleep, mock_openai):
        """Test that large chunk lists are processed in batches."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3]) for _ in range(5)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator(api_key="test_key", batch_size=5)

        chunks = [
            {"chunk_text": f"chunk {i}", "chunk_index": i, "url": "cv://sample-candidate", "title": "Test",
             "metadata": {}}
            for i in range(12)
        ]

        result = generator.generate_embeddings(chunks)

        assert mock_client.embeddings.create.call_count == 3
        assert len(result) == 12

    @patch('processors.embedding_generator.OpenAI')
    def test_generate_single_embedding(self, mock_openai):
        """Test single text embedding generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator(api_key="test_key")

        embedding = generator.generate_single_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            input=["test text"],
            model="text-embedding-3-small"
        )

    @patch('processors.embedding_generator.OpenAI')
    def test_custom_model(self, mock_openai):
        """Test using custom embedding model."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator(
            api_key="test_key",
            model="text-embedding-ada-002"
        )

        assert generator.model == "text-embedding-ada-002"
