"""
Unit tests for chat API service.

Tests the ChatService class with mocked ChatEngine, DocumentProcessor, and database session.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.api.chat_service import ChatService
from app.api.dto import ChatRequest, ChatResponse, ConversationResponse
from app.models.conversation import Conversation
from app.models.document import Document


@pytest.fixture
def chat_service():
    """Create a ChatService instance."""
    with patch('app.api.chat_service.ChatEngine'):
        return ChatService()


class TestSendMessage:
    """Test cases for ChatService.send_message method."""

    @pytest.mark.asyncio
    async def test_send_message_new_conversation(
        self,
        chat_service,
        db_session,
        sample_document
    ):
        """Test sending a message in a new conversation."""
        # Arrange
        request = ChatRequest(
            message="What is the main topic?",
            conversation_id=None,
            document_ids=[sample_document.id]
        )
        
        chat_service.chat_engine.process_message = AsyncMock(
            return_value={
                "answer": "The main topic is machine learning.",
                "sources": [{"id": 1, "type": "text", "page": 1}],
                "processing_time": 1.5
            }
        )
        
        # Act
        with patch('app.api.chat_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            result = await chat_service.send_message(request)
        
        # Assert
        assert isinstance(result, ChatResponse)
        assert result.conversation_id is not None
        assert result.message_id is not None
        assert result.answer == "The main topic is machine learning."
        assert result.processing_time == 1.5

    @pytest.mark.asyncio
    async def test_send_message_existing_conversation(
        self,
        chat_service,
        db_session,
        sample_conversation,
        sample_document
    ):
        """Test sending a message to an existing conversation."""
        # Arrange
        request = ChatRequest(
            message="Tell me more",
            conversation_id=sample_conversation.id,
            document_ids=[sample_document.id]
        )
        
        chat_service.chat_engine.process_message = AsyncMock(
            return_value={
                "answer": "Here are more details.",
                "sources": [],
                "processing_time": 0.8
            }
        )
        
        # Act
        with patch('app.api.chat_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            result = await chat_service.send_message(request)
        
        # Assert
        assert result.conversation_id == sample_conversation.id

    @pytest.mark.asyncio
    async def test_send_message_with_multiple_documents(
        self,
        chat_service,
        db_session
    ):
        """Test sending a message with multiple documents."""
        # Arrange - Create multiple documents
        doc1 = Document(filename="doc1.pdf", file_path="/path/doc1.pdf", processing_status="completed")
        doc2 = Document(filename="doc2.pdf", file_path="/path/doc2.pdf", processing_status="completed")
        db_session.add(doc1)
        db_session.add(doc2)
        db_session.commit()
        db_session.refresh(doc1)
        db_session.refresh(doc2)
        
        request = ChatRequest(
            message="Compare documents",
            conversation_id=None,
            document_ids=[doc1.id, doc2.id]
        )
        
        chat_service.chat_engine.process_message = AsyncMock(
            return_value={
                "answer": "Document comparison results.",
                "sources": [],
                "processing_time": 2.0
            }
        )
        
        # Act
        with patch('app.api.chat_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            result = await chat_service.send_message(request)
        
        # Assert
        assert result.answer == "Document comparison results."

    @pytest.mark.asyncio
    async def test_send_message_document_not_found(
        self,
        chat_service,
        db_session
    ):
        """Test sending a message with non-existent document."""
        # Arrange
        request = ChatRequest(
            message="Question",
            conversation_id=None,
            document_ids=[9999]  # Non-existent document
        )
        
        # Act & Assert
        with patch('app.api.chat_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            with pytest.raises(HTTPException) as exc_info:
                await chat_service.send_message(request)
            
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_send_message_chat_engine_error(
        self,
        chat_service,
        db_session,
        sample_document
    ):
        """Test handling of ChatEngine errors."""
        # Arrange
        request = ChatRequest(
            message="Question",
            conversation_id=None,
            document_ids=[sample_document.id]
        )
        
        chat_service.chat_engine.process_message = AsyncMock(
            side_effect=Exception("LLM service error")
        )
        
        # Act
        with patch('app.api.chat_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            result = await chat_service.send_message(request)
        
        # Assert - should return error message instead of raising
        assert "error" in result.answer.lower()
        assert result.sources == []


class TestListConversations:
    """Test cases for ChatService.list method."""

    @pytest.mark.asyncio
    async def test_list_conversations_success(
        self,
        chat_service,
        db_session,
        sample_conversation
    ):
        """Test listing conversations successfully."""
        # Act
        with patch('app.api.chat_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            result = await chat_service.list(skip=0, limit=10)
        
        # Assert
        assert result.total >= 1
        assert result.offset == 0
        assert result.limit == 10

    @pytest.mark.asyncio
    async def test_list_conversations_with_pagination(
        self,
        chat_service,
        db_session
    ):
        """Test listing conversations with pagination."""
        # Act
        with patch('app.api.chat_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            result = await chat_service.list(skip=5, limit=10)
        
        # Assert
        assert result.offset == 5
        assert result.limit == 10

    @pytest.mark.asyncio
    async def test_list_conversations_empty(self, chat_service):
        """Test listing when no conversations exist."""
        # Act
        with patch('app.api.chat_service.scoped_session') as mock_scoped_session:
            db_session = MagicMock()
            mock_scoped_session.return_value.__enter__.return_value = db_session
            
            # Mock the scalar count and execute methods
            db_session.scalar.return_value = 0
            db_session.execute.return_value.scalars.return_value.all.return_value = []
            
            result = await chat_service.list(skip=0, limit=10)
        
        # Assert
        assert result.total == 0
        assert len(result.items) == 0


class TestGetConversation:
    """Test cases for ChatService.get method."""

    @pytest.mark.asyncio
    async def test_get_conversation_success(
        self,
        chat_service,
        db_session,
        sample_conversation
    ):
        """Test getting a conversation successfully."""
        # Act
        with patch('app.api.chat_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            result = await chat_service.get(sample_conversation.id)
        
        # Assert
        assert isinstance(result, ConversationResponse)
        assert result.id == sample_conversation.id

    @pytest.mark.asyncio
    async def test_get_conversation_not_found(
        self,
        chat_service,
        db_session
    ):
        """Test getting a non-existent conversation."""
        # Act & Assert
        with patch('app.api.chat_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            
            with pytest.raises(HTTPException) as exc_info:
                await chat_service.get(9999)
            
            assert exc_info.value.status_code == 404


class TestDeleteConversation:
    """Test cases for ChatService.delete method."""

    @pytest.mark.asyncio
    async def test_delete_conversation_success(
        self,
        chat_service: ChatService,
        db_session: Session,
        sample_conversation: Conversation,
    ):
        """Test deleting a conversation successfully."""
        # Act
        with patch('app.api.chat_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            await chat_service.delete(sample_conversation.id)
        
        assert db_session.get(Conversation, sample_conversation.id) is None

    @pytest.mark.asyncio
    async def test_delete_conversation_not_found(
        self,
        chat_service: ChatService,
        db_session: Session,
    ):
        """Test deleting a non-existent conversation."""
        # Act & Assert
        with patch('app.api.chat_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            
            with pytest.raises(HTTPException) as exc_info:
                await chat_service.delete(9999)
            
            assert exc_info.value.status_code == 404
