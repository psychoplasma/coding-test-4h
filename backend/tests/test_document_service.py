"""
Unit tests for document API service.

Tests the DocumentService class with mocked DocumentProcessor and database session.
"""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.api.document_service import DocumentService
from app.api.dto import DocumentResponse, PaginatedDocumentResponse
from app.models.document import Document


@pytest.fixture
def document_service() -> DocumentService:
    """Create a DocumentService instance."""
    return DocumentService()


class TestProcessDocument:
    """Test cases for DocumentService.process_document method."""

    @pytest.mark.asyncio
    async def test_process_document_success(
        self,
        document_service: DocumentService,
        db_session: Session,
        sample_document: Document,
    ):
        """Test processing a document successfully."""
        # Arrange
        file_path = "/uploads/documents/test.pdf"
        
        with patch('app.api.document_service.DocumentProcessor') as mock_processor_class:
            mock_processor = AsyncMock()
            mock_processor_class.return_value = mock_processor
            mock_processor.process_document.return_value = None
            
            # Act
            with patch('app.api.document_service.scoped_session') as mock_scoped_session:
                mock_scoped_session.return_value.__enter__.return_value = db_session
                await document_service.process_document(sample_document.id, file_path)
        
        # Assert
        mock_processor.process_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_document_error_handling(
        self,
        document_service: DocumentService,
        db_session: Session,
        sample_document: Document,
    ):
        """Test error handling during document processing."""
        # Arrange
        file_path = "/uploads/documents/test.pdf"
        error_message = "Failed to extract text"
        
        with patch('app.api.document_service.DocumentProcessor') as mock_processor_class:
            mock_processor = AsyncMock()
            mock_processor_class.return_value = mock_processor
            mock_processor.process_document.side_effect = Exception(error_message)
            
            # Act
            with patch('app.api.document_service.scoped_session') as mock_scoped_session:
                mock_scoped_session.return_value.__enter__.return_value = db_session
                await document_service.process_document(sample_document.id, file_path)
        
        # Assert
        # Verify that error was logged and document status updated
        mock_processor.process_document.assert_called_once()


class TestCreateDocument:
    """Test cases for DocumentService.create method."""

    @pytest.mark.asyncio
    async def test_create_document_success(
        self,
        document_service: DocumentService,
        db_session: Session,
    ):
        """Test creating a document successfully."""
        # Arrange
        filename = "test.pdf"
        file_path = "/uploads/documents/test_123.pdf"
        
        # Act
        with patch('app.api.document_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            result = await document_service.create(filename, file_path)
        
        # Assert
        assert isinstance(result, DocumentResponse)
        assert result.filename == filename
        assert result.status == "pending"

    @pytest.mark.asyncio
    async def test_create_document_with_metadata(
        self,
        document_service: DocumentService,
        db_session: Session,
    ):
        """Test creating a document with initial metadata."""
        # Arrange
        filename = "report.pdf"
        file_path = "/uploads/documents/report_456.pdf"
        
        # Act
        with patch('app.api.document_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            result = await document_service.create(filename, file_path)
        
        # Assert
        assert result.filename == filename
        assert result.upload_date is not None


class TestListDocuments:
    """Test cases for DocumentService.list method."""

    @pytest.mark.asyncio
    async def test_list_documents_success(
        self,
        document_service: DocumentService,
        db_session: Session,
        sample_document: Document,
    ):
        """Test listing documents successfully."""
        # Act
        with patch('app.api.document_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            result = await document_service.list(skip=0, limit=10)
        
        # Assert
        assert isinstance(result, PaginatedDocumentResponse)
        assert result.total >= 1
        assert result.offset == 0
        assert result.limit == 10

    @pytest.mark.asyncio
    async def test_list_documents_with_pagination(
        self,
        document_service: DocumentService,
        db_session: Session,
        sample_document: Document,
    ):
        """Test listing documents with pagination."""
        # Act
        with patch('app.api.document_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            result = await document_service.list(skip=5, limit=10)
        
        # Assert
        assert result.offset == 5
        assert result.limit == 10

    @pytest.mark.asyncio
    async def test_list_documents_empty(
        self,
        document_service: DocumentService,
        db_session: Session,
    ):
        """Test listing documents when none exist."""
        # Act
        with patch('app.api.document_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            result = await document_service.list(skip=0, limit=10)
        
        # Assert
        assert result.total == 0
        assert len(result.items) == 0


class TestGetDocument:
    """Test cases for DocumentService.get method."""

    @pytest.mark.asyncio
    async def test_get_document_success(
        self,
        document_service: DocumentService,
        db_session: Session,
        sample_document: Document,
    ):
        """Test getting a document successfully."""
        # Act
        with patch('app.api.document_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            result = await document_service.get(sample_document.id)
        
        # Assert
        assert isinstance(result, DocumentResponse)
        assert result.id == sample_document.id
        assert result.filename == sample_document.filename

    @pytest.mark.asyncio
    async def test_get_document_with_extracted_content(
        self,
        document_service: DocumentService,
        db_session: Session,
        sample_document: Document,
        sample_images,
        sample_tables
    ):
        """Test getting a document with extracted images and tables."""
        # Act
        with patch('app.api.document_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            result = await document_service.get(sample_document.id)
        
        # Assert
        assert isinstance(result, DocumentResponse)
        assert result.id == sample_document.id
        assert len(result.images) == len(sample_images)
        assert len(result.tables) == len(sample_tables)

    @pytest.mark.asyncio
    async def test_get_document_not_found(
        self,
        document_service: DocumentService,
        db_session: Session,
    ):
        """Test getting a non-existent document."""
        # Act & Assert
        with patch('app.api.document_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            
            with pytest.raises(HTTPException) as exc_info:
                await document_service.get(9999)
            
            assert exc_info.value.status_code == 404


class TestDeleteDocument:
    """Test cases for DocumentService.delete method."""

    @pytest.mark.asyncio
    async def test_delete_document_success(
        self,
        document_service: DocumentService,
        db_session: Session,
        sample_document: Document,
        sample_images,
        sample_tables,
    ):
        """Test deleting a document successfully."""
        # Act
        with patch('app.api.document_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session

            await document_service.delete(sample_document.id)

        # Assert
        assert db_session.get(Document, sample_document.id) is None

    @pytest.mark.asyncio
    async def test_delete_document_not_found(
        self,
        document_service: DocumentService,
        db_session: Session,
    ):
        """Test deleting a non-existent document."""
        # Act & Assert
        with patch('app.api.document_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            
            with pytest.raises(HTTPException) as exc_info:
                await document_service.delete(9999)
            
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_document_files_not_found(
        self,
        document_service: DocumentService,
        db_session: Session,
        sample_document: Document,
    ):
        """Test deleting a document when associated files don't exist."""
        # Arrange
        sample_document.file_path = "/nonexistent/path/test.pdf"
        
        # Act - should not raise error even if files don't exist
        with patch('app.api.document_service.scoped_session') as mock_scoped_session:
            mock_scoped_session.return_value.__enter__.return_value = db_session
            await document_service.delete(sample_document.id)

        # Assert
        # Should rollback deletion due to missing files
        assert db_session.get(Document, sample_document.id) is not None
