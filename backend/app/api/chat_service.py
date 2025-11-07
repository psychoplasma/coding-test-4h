"""
Chat service API layer
"""
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import scoped_session
from app.models.conversation import Conversation, Message
from app.services.chat_engine import ChatEngine
from backend.app.api.chat_controller import _get_conversation_by_id
from app.api.dto import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    PaginatedConversationResponse,
)


class ChatService():
    def __init__(self):
        self.chat_engine = ChatEngine()

    async def send_message(self, request: ChatRequest) -> ChatResponse:
        """
        Send a chat message and get a response
        
        This endpoint:
        1. Creates or retrieves conversation
        2. Saves user message
        3. Processes message with ChatEngine (RAG + multimodal)
        4. Saves assistant response
        5. Returns answer with sources (text, images, tables)
        """
        with scoped_session() as session:
            # Create or get conversation
            if request.conversation_id:
                conversation = _get_conversation_by_id(
                    session,
                    request.conversation_id
                )
            else:
                conversation = Conversation(
                    title=request.message[:50],  # First 50 chars as title
                    document_id=request.document_id
                )
                session.add(conversation)
                session.flush()  # Flush to get id

            # Save user message
            user_message = Message(
                conversation_id=conversation.id,
                role="user",
                content=request.message
            )
            session.add(user_message)
            session.flush() # Flush to get id

            try:
                # Process message with ChatEngine
                result = await self.chat_engine.process_message(
                    session=session,
                    conversation_id=conversation.id,
                    message=request.message,
                    document_ids=request.document_ids
                )
            except Exception as e:
                result = {
                    "answer": f"Sorry, I encountered an error processing your message: {str(e)}",
                    "sources": [],
                    "processing_time": 0.0
                }

            # Save assistant message
            assistant_message = Message(
                conversation_id=conversation.id,
                role="assistant",
                content=result["answer"],
                sources=result.get("sources", [])
            )
            session.add(assistant_message)
            session.flush() # Flush to get id

            return ChatResponse(
                conversation_id=conversation.id,
                message_id=assistant_message.id,
                answer=result["answer"],
                sources=result.get("sources", []),
                processing_time=result.get("processing_time", 0.0)
            )

    async def list(
        self,
        skip: int = 0,
        limit: int = 10,
    ) -> PaginatedConversationResponse:
        """
        Get list of all conversations
        """
        with scoped_session() as session:
            total = session.scalar(select(func.count()).select_from(Conversation))
            conversations = session.execute(
                select(Conversation)
                .offset(skip)
                .limit(limit)
            ).scalars().all()

            return PaginatedConversationResponse(
                total=total,
                offset=skip,
                limit=limit,
                items=[ConversationResponse.from_model(c) for c in conversations],
            )

    async def get(self, conversation_id: int) -> ConversationResponse:
        """
        Get conversation details with all messages
        """
        with scoped_session() as session:
            conversation = self._get_conversation_by_id(
                session,
                conversation_id,
            )

            return ConversationResponse.from_model(conversation)

    async def delete(
        self,
        conversation_id: int,
    ) -> None:
        """
        Delete a conversation and all its messages
        """
        with scoped_session() as session:
            conversation = self._get_conversation_by_id(
                session,
                conversation_id,
            )
            session.delete(conversation)
            session.commit()

    def _get_conversation_by_id(
        self,
        session: Session,
        conversation_id: int
    ) -> Conversation:
        conversation = session.get(Conversation, conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
