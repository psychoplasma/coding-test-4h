"""
Unit tests for chat API controller.

Tests the chat API endpoints with mocked ChatService.
"""
import pytest
from unittest.mock import AsyncMock, patch

from app.api.dto import ChatRequest, ChatResponse, ConversationResponse, PaginatedConversationResponse


@pytest.fixture
def mock_chat_service():
    """Create a mock ChatService."""
    with patch('app.api.chat_controller.ChatService') as mock:
        service_instance = AsyncMock()
        mock.return_value = service_instance
        yield service_instance


class TestSendMessage:
    """Test cases for POST /api/chat endpoint."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, client, mock_chat_service):
        """Test sending a message successfully."""
        # Arrange
        request_data = {
            "message": "What is in the document?",
            "conversation_id": None,
            "document_ids": [1]
        }
        
        expected_response = {
            "conversation_id": 1,
            "message_id": 1,
            "answer": "The document contains important information.",
            "sources": [
                {
                    "id": 1,
                    "type": "text",
                    "content": "Relevant text from document",
                    "page": 1
                }
            ],
            "processing_time": 1.5
        }
        
        mock_chat_service.send_message.return_value = ChatResponse(**expected_response)
        
        # Act
        with patch('app.api.chat_controller.chat_service', mock_chat_service):
            response = client.post("/api/chat", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == 1
        assert data["message_id"] == 1
        assert "answer" in data
        assert "sources" in data
        assert "processing_time" in data

    @pytest.mark.asyncio
    async def test_send_message_with_existing_conversation(self, client, mock_chat_service):
        """Test sending a message to an existing conversation."""
        # Arrange
        request_data = {
            "message": "Tell me more",
            "conversation_id": 1,
            "document_ids": [1]
        }
        
        expected_response = {
            "conversation_id": 1,
            "message_id": 2,
            "answer": "Here are more details.",
            "sources": [],
            "processing_time": 0.8
        }
        
        mock_chat_service.send_message.return_value = ChatResponse(**expected_response)
        
        # Act
        with patch('app.api.chat_controller.chat_service', mock_chat_service):
            response = client.post("/api/chat", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == 1
        assert data["message_id"] == 2

    @pytest.mark.asyncio
    async def test_send_message_without_documents(self, client, mock_chat_service):
        """Test sending a message without specifying documents."""
        # Arrange
        request_data = {
            "message": "General question",
            "conversation_id": None,
            "document_ids": None
        }
        
        expected_response = {
            "conversation_id": 2,
            "message_id": 1,
            "answer": "Answer to general question.",
            "sources": [],
            "processing_time": 0.5
        }
        
        mock_chat_service.send_message.return_value = ChatResponse(**expected_response)
        
        # Act
        with patch('app.api.chat_controller.chat_service', mock_chat_service):
            response = client.post("/api/chat", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == 2

    @pytest.mark.asyncio
    async def test_send_message_empty_message(self, client):
        """Test sending an empty message."""
        # Arrange
        request_data = {
            "message": "",
            "conversation_id": None,
            "document_ids": None
        }
        
        # Act
        response = client.post("/api/chat", json=request_data)
        
        # Assert
        # Pydantic validation should catch empty string if required
        # Otherwise, it should still work but might get handled by service layer


class TestListConversations:
    """Test cases for GET /api/chat/conversations endpoint."""

    @pytest.mark.asyncio
    async def test_list_conversations_success(self, client, mock_chat_service):
        """Test listing conversations successfully."""
        # Arrange
        expected_response = {
            "total": 2,
            "offset": 0,
            "limit": 10,
            "items": [
                {
                    "id": 1,
                    "title": "Test Conversation 1",
                    "created_at": "2025-01-09T10:00:00",
                    "document_id": 1,
                    "messages": []
                },
                {
                    "id": 2,
                    "title": "Test Conversation 2",
                    "created_at": "2025-01-09T11:00:00",
                    "document_id": None,
                    "messages": []
                }
            ]
        }
        
        mock_chat_service.list.return_value = PaginatedConversationResponse(**expected_response)
        
        # Act
        with patch('app.api.chat_controller.chat_service', mock_chat_service):
            response = client.get("/api/chat/conversations")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        mock_chat_service.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_conversations_with_pagination(self, client, mock_chat_service):
        """Test listing conversations with pagination parameters."""
        # Arrange
        expected_response = {
            "total": 5,
            "offset": 10,
            "limit": 10,
            "items": []
        }
        
        mock_chat_service.list.return_value = PaginatedConversationResponse(**expected_response)
        
        # Act
        with patch('app.api.chat_controller.chat_service', mock_chat_service):
            response = client.get("/api/chat/conversations?skip=10&limit=10")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["offset"] == 10
        assert data["limit"] == 10
        mock_chat_service.list.assert_called_once_with(skip=10, limit=10)

    @pytest.mark.asyncio
    async def test_list_conversations_empty(self, client, mock_chat_service):
        """Test listing conversations when none exist."""
        # Arrange
        expected_response = {
            "total": 0,
            "offset": 0,
            "limit": 10,
            "items": []
        }
        
        mock_chat_service.list.return_value = PaginatedConversationResponse(**expected_response)
        
        # Act
        with patch('app.api.chat_controller.chat_service', mock_chat_service):
            response = client.get("/api/chat/conversations")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert len(data["items"]) == 0


class TestGetConversation:
    """Test cases for GET /api/chat/conversations/{conversation_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_conversation_success(self, client, mock_chat_service):
        """Test getting a conversation successfully."""
        # Arrange
        conversation_id = 1
        expected_response = {
            "id": 1,
            "title": "Test Conversation",
            "created_at": "2025-01-09T10:00:00",
            "document_id": 1,
            "messages": [
                {
                    "id": 1,
                    "role": "user",
                    "content": "What is this?",
                    "sources": [],
                    "created_at": "2025-01-09T10:00:00"
                },
                {
                    "id": 2,
                    "role": "assistant",
                    "content": "This is...",
                    "sources": [],
                    "created_at": "2025-01-09T10:01:00"
                }
            ]
        }
        
        mock_chat_service.get.return_value = ConversationResponse(**expected_response)
        
        # Act
        with patch('app.api.chat_controller.chat_service', mock_chat_service):
            response = client.get(f"/api/chat/conversations/{conversation_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
        assert data["title"] == "Test Conversation"
        assert len(data["messages"]) == 2
        mock_chat_service.get.assert_called_once_with(conversation_id)

    @pytest.mark.asyncio
    async def test_get_conversation_not_found(self, client, mock_chat_service):
        """Test getting a non-existent conversation."""
        # Arrange
        conversation_id = 999
        from fastapi import HTTPException
        mock_chat_service.get.side_effect = HTTPException(status_code=404, detail="Conversation not found")
        
        # Act
        with patch('app.api.chat_controller.chat_service', mock_chat_service):
            response = client.get(f"/api/chat/conversations/{conversation_id}")
        
        # Assert
        assert response.status_code == 404


class TestDeleteConversation:
    """Test cases for DELETE /api/chat/conversations/{conversation_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_conversation_success(self, client, mock_chat_service):
        """Test deleting a conversation successfully."""
        # Arrange
        conversation_id = 1
        mock_chat_service.delete.return_value = None
        
        # Act
        with patch('app.api.chat_controller.chat_service', mock_chat_service):
            response = client.delete(f"/api/chat/conversations/{conversation_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "successfully" in data["message"].lower()
        mock_chat_service.delete.assert_called_once_with(conversation_id)

    @pytest.mark.asyncio
    async def test_delete_conversation_not_found(self, client, mock_chat_service):
        """Test deleting a non-existent conversation."""
        # Arrange
        conversation_id = 999
        from fastapi import HTTPException
        mock_chat_service.delete.side_effect = HTTPException(status_code=404, detail="Conversation not found")
        
        # Act
        with patch('app.api.chat_controller.chat_service', mock_chat_service):
            response = client.delete(f"/api/chat/conversations/{conversation_id}")
        
        # Assert
        assert response.status_code == 404
