"""
Document management API endpoints
"""
from logging import getLogger
import os

from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.db.session import scoped_session
from app.models.document import Document
from app.services.document_processor import DocumentProcessor
from app.api.dto import (
    DocumentResponse,
    PaginatedDocumentResponse,
)


logger = getLogger(__name__)


class DocumentService:
    """Service for managing documents"""

    def __init__(self):
        pass

    async def process_document(self, document_id: int, file_path: str) -> None:
        """Background task to process the uploaded document"""
        try:
            with scoped_session() as session:
                processor = DocumentProcessor()
                await processor.process_document(session, file_path, document_id)
        except Exception as e:
            logger.error(f"Error processing document ID {document_id}: {e}")
            with scoped_session() as session:
                document = self._get(session, document_id)
                document.processing_status = "failed"
                document.error_message = str(e)

    async def create(self, filename: str, file_path: str) -> DocumentResponse:
        """
        Create a new document record
        """
        with scoped_session() as session:
            document = Document(
                filename=filename,
                file_path=file_path,
                processing_status="pending",
            )
            session.add(document)
            session.flush() # To get the document ID

            return DocumentResponse.from_model(document)

    async def list(self, skip: int = 0, limit: int = 10) -> PaginatedDocumentResponse:
        """
        Get list of all documents
        """
        with scoped_session() as session:
            total = session.scalar(select(func.count()).select_from(Document))
            documents = session.execute(
                select(Document)
                .offset(skip)
                .limit(limit)
            ).scalars().all()

            return PaginatedDocumentResponse(
                total=total,
                offset=skip,
                limit=limit,
                items=[DocumentResponse.from_model(doc) for doc in documents]
            )

    async def get(self, document_id: int) -> DocumentResponse:
        """
        Get document details including extracted images and tables
        """
        with scoped_session() as session:
            document = self._get(session, document_id)

            return DocumentResponse.from_model(document)

    async def delete(self, document_id: int) -> None:
        """
        Delete a document and all associated data
        """
        with scoped_session() as session:
            document = self._get(session, document_id)

            # Delete physical files
            # If document file doesn't exist, skip deletion to maintain data integrity
            if not os.path.exists(document.file_path):
                logger.warning(f"Document file not found: {document.file_path}. Skipping deletion to maintain data integrity.")
                return

            try:
                os.remove(document.file_path)
            except Exception as e:
                logger.warning(f"Failed to delete document file {document.file_path}: {e}")

            for img in document.images:
                try:
                    if os.path.exists(img.file_path):
                        os.remove(img.file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete image file {img.file_path}: {e}")

            for tbl in document.tables:
                try:
                    if os.path.exists(tbl.image_path):
                        os.remove(tbl.image_path)
                except Exception as e:
                    logger.warning(f"Failed to delete table file {tbl.image_path}: {e}")

            # Delete database record (cascade will handle related records)
            session.delete(document)
            session.flush()  # Ensure deletion is persisted

    def _get(self, session: Session, document_id: int) -> Document:
        document = session.get(Document, document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
        return document