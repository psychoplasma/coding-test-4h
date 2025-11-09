
import os
from typing import Optional, List
from pydantic import BaseModel, Field

from app.models.document import Document


class ChatRequest(BaseModel):
    """Request model for sending a chat message"""
    message: str = Field(..., description="The user's message to the chat service")
    conversation_id: Optional[int] = Field(None, description="ID of the existing conversation")
    document_ids: Optional[List[int]] = Field(None, description="IDs of the documents being referenced")

class ChatResponse(BaseModel):
    """Response model for chat message"""
    conversation_id: int = Field(..., description="ID of the conversation")
    message_id: int = Field(..., description="ID of the message")
    answer: str = Field(..., description="The assistant's answer to the user's message")
    sources: List[dict] = Field(..., description="List of sources (text, images, tables) used in the answer")
    processing_time: float = Field(..., description="Time taken to process the request")

class MessageResponse(BaseModel):
    """Response model for individual messages"""
    id: int = Field(..., description="ID of the message")
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")
    sources: List[dict] = Field(..., description="List of sources (text, images, tables) used in the answer")
    created_at: str = Field(..., description="Creation timestamp of the message")

    @staticmethod
    def from_model(message) -> "MessageResponse":
        return MessageResponse(
            id=message.id,
            role=message.role,
            content=message.content,
            sources=message.sources,
            created_at=message.created_at.isoformat()
        )

class ConversationResponse(BaseModel):
    """Response model for conversation details"""
    id: int = Field(..., description="ID of the conversation")
    title: str = Field(..., description="Title of the conversation")
    created_at: str = Field(..., description="Creation timestamp of the conversation")
    document_id: Optional[int] = Field(None, description="ID of the associated document")
    messages: List[dict] = Field(..., description="List of messages in the conversation")

    @staticmethod
    def from_model(conversation) -> "ConversationResponse":
        return ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at.isoformat(),
            document_id=conversation.document_id,
            messages=[MessageResponse.from_model(message) for message in conversation.messages]
        )

class PaginatedConversationResponse(BaseModel):
    """Paginated response model for conversations"""
    total: int = Field(..., description="Total number of conversations")
    offset: int = Field(..., description="Offset of the current page")
    limit: int = Field(..., description="Limit of items per page")
    items: List[ConversationResponse] = Field(..., description="List of conversations in the current page")

class DocumentResponse(BaseModel):
    """Response model for document details"""
    id: int = Field(..., description="ID of the document")
    filename: str = Field(..., description="Filename of the document")
    upload_date: str = Field(..., description="Upload date of the document")
    status: str = Field(..., description="Processing status of the document")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    total_pages: int = Field(..., description="Total number of pages in the document")
    text_chunks: int = Field(..., description="Number of text chunks extracted from the document")
    images: List[dict] = Field(..., description="List of extracted images from the document")
    tables: List[dict] = Field(..., description="List of extracted tables from the document")

    @staticmethod
    def from_model(document: Document) -> "DocumentResponse":
        return DocumentResponse(
            id=document.id,
            filename=document.filename,
            upload_date=document.upload_date.isoformat(),
            status=document.processing_status,
            error_message=document.error_message,
            total_pages=document.total_pages,
            text_chunks=document.text_chunks_count,
            images=[
                {
                    "id": img.id,
                    "url": f"/uploads/images/{os.path.basename(img.file_path)}",
                    "page": img.page_number,
                    "caption": img.caption,
                    "width": img.width,
                    "height": img.height
                }
                for img in document.images
            ],
            tables=[
                {
                    "id": tbl.id,
                    "url": f"/uploads/tables/{os.path.basename(tbl.image_path)}",
                    "page": tbl.page_number,
                    "caption": tbl.caption,
                    "rows": tbl.rows,
                    "columns": tbl.columns,
                    "data": tbl.data
                }
                for tbl in document.tables
            ],
        )

class PaginatedDocumentResponse(BaseModel):
    """Paginated response model for documents"""
    total: int = Field(..., description="Total number of documents")
    offset: int = Field(..., description="Offset of the current page")
    limit: int = Field(..., description="Limit of items per page")
    items: List[DocumentResponse] = Field(..., description="List of documents in the current page")

class UploadDocumentResponse(BaseModel):
    """Response model for document upload"""
    id: int = Field(..., description="ID of the uploaded document")
    filename: str = Field(..., description="Filename of the uploaded document")
    status: str = Field(..., description="Processing status of the uploaded document")
    message: str = Field(..., description="Informational message about the upload")

class DeleteResponse(BaseModel):
    """Response model for delete operations"""
    message: str = Field(..., description="Informational message about the delete operation")