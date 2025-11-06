"""
Chat API endpoints
"""
from fastapi import APIRouter
from app.api.chat_service import ChatService
from app.api.dto import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    ConversationResponse,
    PaginatedConversationResponse,
)

router = APIRouter()

chat_service = ChatService()

@router.post("", response_model=ChatResponse)
async def send_message(request: ChatRequest) -> ChatResponse:
    """
    Send a chat message and get a response
    
    This endpoint:
    1. Creates or retrieves conversation
    2. Saves user message
    3. Processes message with ChatEngine (RAG + multimodal)
    4. Saves assistant response
    5. Returns answer with sources (text, images, tables)
    """
    return await chat_service.send_message(request)

@router.get("/conversations", response_model=PaginatedConversationResponse)
async def list_conversations(
    skip: int = 0,
    limit: int = 10,
) -> PaginatedConversationResponse:
    """
    Get list of all conversations
    """
    return await chat_service.list(skip=skip, limit=limit)

@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: int):
    """
    Get conversation details with all messages
    """
    return await chat_service.get(conversation_id)

@router.delete("/conversations/{conversation_id}", response_model=DeleteResponse)
async def delete_conversation(conversation_id: int) -> DeleteResponse:
    """
    Delete a conversation and all its messages
    """
    await chat_service.delete(conversation_id)
    return DeleteResponse(message="Conversation deleted successfully")
