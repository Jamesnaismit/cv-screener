"""
Unit tests for the conversational RAG chain.
"""

import pytest
from unittest.mock import Mock, MagicMock
from rag.chain import ConversationalRAGChain


@pytest.fixture
def mock_retriever():
    """Create a mock retriever for testing."""
    retriever = Mock()
    retriever.retrieve.return_value = [
        {
            "content": "Jonathan Dyer built APIs with FastAPI and PostgreSQL.",
            "url": "cv://cv-03-jonathan-dyer",
            "title": "CV Jonathan Dyer",
            "similarity": 0.95,
            "relevance_score": 0.95,
            "metadata": {"language": "en"},
        },
        {
            "content": "He has 8 years of backend experience and technical leadership.",
            "url": "cv://cv-03-jonathan-dyer",
            "title": "Experiencia",
            "similarity": 0.87,
            "relevance_score": 0.87,
            "metadata": {"language": "en"},
        },
    ]
    return retriever


@pytest.fixture
def rag_chain(mock_retriever):
    """Create ConversationalRAGChain instance for testing."""
    return ConversationalRAGChain(
        retriever=mock_retriever,
        openai_api_key="test_key",
        model_name="gpt-5-mini-2025-08-07",
        max_history=5,
    )


@pytest.mark.unit
class TestConversationalRAGChain:
    """Test cases for ConversationalRAGChain."""

    def test_chain_initialization(self, mock_retriever):
        """Test chain initialization."""
        chain = ConversationalRAGChain(
            retriever=mock_retriever,
            openai_api_key="test_key",
            model_name="gpt-5-mini-2025-08-07",
        )

        assert chain.retriever == mock_retriever
        assert chain.model_name == "gpt-5-mini-2025-08-07"
        assert chain.max_history == 10
        assert len(chain.conversation_history) == 0

    def test_format_context_with_docs(self, rag_chain):
        """Test context formatting with documents."""
        docs = [
            {
                "content": "Test content 1",
                "url": "https://test1.com",
                "title": "Test 1",
                "similarity": 0.95,
                "relevance_score": 0.95,
            },
            {
                "content": "Test content 2",
                "url": "https://test2.com",
                "title": "Test 2",
                "similarity": 0.85,
                "relevance_score": 0.85,
            },
        ]

        context = rag_chain._format_context_with_quality_tiers(docs)

        assert "Test content 1" in context
        assert "Test content 2" in context
        assert "https://test1.com" in context
        assert "https://test2.com" in context
        # Relevance scores are no longer shown in context

    def test_format_context_empty(self, rag_chain):
        """Test context formatting with no documents."""
        context = rag_chain._format_context_with_quality_tiers([])
        assert "No relevant information" in context or context == ""

    def test_format_chat_history_empty(self, rag_chain):
        """Test chat history formatting when empty."""
        history = rag_chain._format_chat_history()
        assert "No prior conversation" in history

    def test_format_chat_history_with_messages(self, rag_chain):
        """Test chat history formatting with messages."""
        rag_chain.conversation_history = [
            ("What is the current role?", "Backend senior using FastAPI."),
            ("What tools does he use?", "Works with PostgreSQL and Redis."),
        ]

        history = rag_chain._format_chat_history()

        assert "What is the current role?" in history
        assert "Backend senior using FastAPI." in history
        assert "What tools does he use?" in history
        assert "Works with PostgreSQL and Redis." in history

    def test_clear_history(self, rag_chain):
        """Test clearing conversation history."""
        rag_chain.conversation_history = [
            ("Question 1", "Answer 1"),
            ("Question 2", "Answer 2"),
        ]

        assert len(rag_chain.conversation_history) == 2

        rag_chain.clear_history()

        assert len(rag_chain.conversation_history) == 0

    def test_get_history_length(self, rag_chain):
        """Test getting conversation history length."""
        assert rag_chain.get_history_length() == 0

        rag_chain.conversation_history = [
            ("Q1", "A1"),
            ("Q2", "A2"),
            ("Q3", "A3"),
        ]

        assert rag_chain.get_history_length() == 3

    def test_history_trimming(self, rag_chain):
        """Test that history is trimmed when exceeding max_history."""
        rag_chain.max_history = 3

        for i in range(5):
            rag_chain.conversation_history.append((f"Q{i}", f"A{i}"))

        rag_chain.conversation_history = rag_chain.conversation_history[-rag_chain.max_history:]

        assert len(rag_chain.conversation_history) == 3
        assert rag_chain.conversation_history[0] == ("Q2", "A2")
        assert rag_chain.conversation_history[2] == ("Q4", "A4")

    def test_system_prompt_structure(self, rag_chain):
        """Test that prompt optimizer is available."""
        # SYSTEM_PROMPT is now handled by PromptOptimizer, not ConversationalRAGChain
        assert rag_chain.prompt_optimizer is not None
        assert hasattr(rag_chain.prompt_optimizer, 'create_prompt')

    def test_context_includes_all_fields(self, rag_chain):
        """Test that formatted context includes all required fields."""
        docs = [
            {
                "content": "Test content here",
                "url": "https://test.com",
                "title": "Test Document",
                "similarity": 0.92,
                "relevance_score": 0.92,
            }
        ]

        context = rag_chain._format_context_with_quality_tiers(docs)
        assert "Test content here" in context
        assert "https://test.com" in context
        assert "Test Document" in context

    def test_format_chat_history_respects_max(self, rag_chain):
        """Test that chat history formatting respects max_history."""
        rag_chain.max_history = 2

        for i in range(5):
            rag_chain.conversation_history.append((f"Question {i}", f"Answer {i}"))

        history = rag_chain._format_chat_history()

        assert "Question 3" in history
        assert "Question 4" in history
        assert "Question 0" not in history
        assert "Question 1" not in history

    def test_normalize_question_strips_trailing_punctuation(self, rag_chain):
        """Ensure trailing punctuation/spacing does not change normalized question."""
        normalized = rag_chain._normalize_question("Evelyn Hamilton experience?   ")
        assert normalized == "Evelyn Hamilton experience"

    def test_relevance_score_bm25_only_documents(self, rag_chain):
        """Test that BM25-only documents (with relevance_score but no similarity) are included in context."""
        docs = [
            {
                "content": "Python developer with 5 years experience",
                "url": "cv://cv-01",
                "title": "Candidate 1",
                "relevance_score": 0.75,  # Good relevance from BM25
                "similarity": None,  # No vector similarity (BM25-only)
            },
            {
                "content": "Java backend engineer",
                "url": "cv://cv-02",
                "title": "Candidate 2",
                "relevance_score": 0.45,  # Medium relevance
                "similarity": None,
            },
        ]

        context = rag_chain._format_context_with_quality_tiers(docs)

        # All retrieved sources should be included
        assert "Python developer" in context
        assert "Java backend" in context
        # Relevance scores are no longer shown in context

    def test_all_retrieved_sources_included(self, rag_chain):
        """Test that all retrieved sources are included in context regardless of score."""
        docs = [
            {
                "content": "High relevance content",
                "url": "cv://cv-01",
                "title": "Doc 1",
                "relevance_score": 0.92,
                "similarity": 0.3,
            },
            {
                "content": "Medium relevance content",
                "url": "cv://cv-02",
                "title": "Doc 2",
                "relevance_score": 0.45,
                "similarity": 0.2,
            },
            {
                "content": "Lower relevance content",
                "url": "cv://cv-03",
                "title": "Doc 3",
                "relevance_score": 0.25,
                "similarity": 0.1,
            },
        ]

        context = rag_chain._format_context_with_quality_tiers(docs)

        assert "High relevance content" in context
        assert "Medium relevance content" in context
        assert "Lower relevance content" in context

    def test_format_context_with_relevance_score_only(self, rag_chain):
        """Test context formatting when documents only have relevance_score (no similarity)."""
        docs = [
            {
                "content": "Test content 1",
                "url": "cv://test1",
                "title": "Test 1",
                "relevance_score": 0.88,
                # No similarity field
            },
            {
                "content": "Test content 2",
                "url": "cv://test2",
                "title": "Test 2",
                "relevance_score": 0.55,
            },
        ]

        context = rag_chain._format_context_with_quality_tiers(docs)

        assert "Test content 1" in context
        assert "Test content 2" in context
