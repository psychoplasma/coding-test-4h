"""
Conversation and message models for chat functionality
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.session import Base


# Association table for many-to-many relationship between Conversation and Document to support multiple documents per conversation
class ConversationDocument(Base):
    __tablename__ = 'conversation_documents'

    conversation_id = Column(Integer, ForeignKey('conversations.id', ondelete='CASCADE'), primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), primary_key=True)

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    documents = relationship("Document", secondary="conversation_documents", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)  # Sources used for answer (text, images, tables)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
