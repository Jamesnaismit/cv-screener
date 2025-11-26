"""
Unit tests for the document chunker.
"""

import pytest
from processors.chunker import DocumentChunker


@pytest.fixture
def chunker():
    """Create DocumentChunker instance for testing."""
    return DocumentChunker(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return {
        "url": "cv://sample-candidate",
        "title": "Sample CV",
        "content": "This is a test CV document. " * 50,  # Long enough to split
        "metadata": {"language": "en", "source": "test"},
    }


@pytest.mark.unit
class TestDocumentChunker:
    """Test cases for DocumentChunker."""

    def test_chunker_initialization(self):
        """Test chunker initialization with custom parameters."""
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50
        assert chunker.splitter is not None

    def test_chunker_default_initialization(self):
        """Test chunker initialization with default parameters."""
        chunker = DocumentChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_chunk_document_creates_chunks(self, chunker, sample_document):
        """Test that chunking creates multiple chunks."""
        chunks = chunker.chunk_document(sample_document)

        assert len(chunks) > 1
        assert all(isinstance(chunk, dict) for chunk in chunks)

    def test_chunk_structure(self, chunker, sample_document):
        """Test that chunks have correct structure."""
        chunks = chunker.chunk_document(sample_document)

        for chunk in chunks:
            assert "chunk_text" in chunk
            assert "chunk_index" in chunk
            assert "url" in chunk
            assert "title" in chunk
            assert "metadata" in chunk

            assert chunk["url"] == sample_document["url"]
            assert chunk["title"] == sample_document["title"]

    def test_chunk_indices(self, chunker, sample_document):
        """Test that chunk indices are sequential."""
        chunks = chunker.chunk_document(sample_document)

        for idx, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == idx

    def test_chunk_metadata(self, chunker, sample_document):
        """Test that chunk metadata includes original metadata plus chunk info."""
        chunks = chunker.chunk_document(sample_document)

        for chunk in chunks:
            metadata = chunk["metadata"]

            assert metadata["language"] == "en"
            assert metadata["source"] == "test"

            assert "chunk_index" in metadata
            assert "total_chunks" in metadata
            assert "chunk_size" in metadata
            assert metadata["total_chunks"] == len(chunks)

    def test_empty_content(self, chunker):
        """Test handling of empty content."""
        document = {
            "url": "https://test.com",
            "title": "Empty Doc",
            "content": "",
            "metadata": {},
        }

        chunks = chunker.chunk_document(document)
        assert len(chunks) == 0

    def test_short_content(self, chunker):
        """Test handling of content shorter than chunk size."""
        document = {
            "url": "https://test.com",
            "title": "Short Doc",
            "content": "Short content.",
            "metadata": {},
        }

        chunks = chunker.chunk_document(document)

        assert len(chunks) == 1
        assert chunks[0]["chunk_text"] == "Short content."
        assert chunks[0]["chunk_index"] == 0

    def test_chunk_documents_multiple(self, chunker):
        """Test chunking multiple documents."""
        documents = [
            {
                "url": f"https://test.com/{i}",
                "title": f"Doc {i}",
                "content": "Test content. " * 50,
                "metadata": {},
            }
            for i in range(3)
        ]

        chunks = chunker.chunk_documents(documents)

        assert len(chunks) > 3

        urls = {chunk["url"] for chunk in chunks}
        assert len(urls) == 3

    def test_chunk_size_respected(self, chunker, sample_document):
        """Test that chunks respect max chunk size."""
        chunks = chunker.chunk_document(sample_document)

        for chunk in chunks:
            assert len(chunk["chunk_text"]) <= chunker.chunk_size + 50  # Allow margin

    def test_chunk_documents_handles_errors(self, chunker):
        """Test that chunking continues even if one document fails."""
        documents = [
            {"url": "https://test1.com", "title": "Doc 1", "content": "Content 1" * 50, "metadata": {}},
            {"url": "https://test2.com", "title": "Doc 2"},  # Missing content
            {"url": "https://test3.com", "title": "Doc 3", "content": "Content 3" * 50, "metadata": {}},
        ]

        chunks = chunker.chunk_documents(documents)

        assert len(chunks) > 0
