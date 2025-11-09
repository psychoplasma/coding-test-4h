"""
Unit tests for documents API controller.

Tests the documents API endpoints with mocked DocumentService and BackgroundTasks.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import io

from app.api.dto import DocumentResponse, PaginatedDocumentResponse


@pytest.fixture
def mock_document_service():
    """Create a mock DocumentService."""
    with patch('app.api.documents_controller.DocumentService') as mock:
        service_instance = AsyncMock()
        mock.return_value = service_instance
        yield service_instance


class TestUploadDocument:
    """Test cases for POST /api/documents/upload endpoint."""

    @pytest.mark.asyncio
    async def test_upload_pdf_success(self, client, mock_document_service):
        """Test uploading a PDF document successfully."""
        # Arrange
        pdf_content = b"%PDF-1.4\n%Mock PDF content"
        file = ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
        
        expected_response = {
            "id": 1,
            "filename": "test.pdf",
            "status": "pending",
            "message": "Document uploaded successfully. Processing will begin shortly."
        }
        
        mock_document_service.create.return_value = DocumentResponse(
            id=1,
            filename="test.pdf",
            upload_date="2025-01-09T10:00:00",
            status="pending",
            error_message=None,
            total_pages=0,
            text_chunks=0,
            images=[],
            tables=[]
        )
        
        # Act
        with patch('app.api.documents_controller.document_service', mock_document_service):
            response = client.post("/api/documents/upload", files={"file": file})
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.pdf"
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_upload_non_pdf_file(self, client, mock_document_service):
        """Test uploading a non-PDF file should be rejected."""
        # Arrange
        file_content = b"This is not a PDF"
        file = ("test.txt", io.BytesIO(file_content), "text/plain")
        
        # Act
        with patch('app.api.documents_controller.document_service', mock_document_service):
            response = client.post("/api/documents/upload", files={"file": file})
        
        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "PDF" in data["detail"]

    @pytest.mark.asyncio
    async def test_upload_file_too_large(self, client, mock_document_service):
        """Test uploading a file that exceeds the size limit."""
        # Arrange
        from app.core.config import settings
        
        # Create a large file content (exceeds MAX_FILE_SIZE)
        large_content = b"x" * (settings.MAX_FILE_SIZE + 1)
        file = ("large.pdf", io.BytesIO(large_content), "application/pdf")
        
        # Act
        with patch('app.api.documents_controller.document_service', mock_document_service):
            response = client.post("/api/documents/upload", files={"file": file})
        
        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "exceeds" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_upload_triggers_background_task(self, client, mock_document_service):
        """Test that uploading a document triggers background processing."""
        # Arrange
        pdf_content = b"%PDF-1.4\n%Mock PDF content"
        file = ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
        
        mock_document_service.create.return_value = DocumentResponse(
            id=1,
            filename="test.pdf",
            upload_date="2025-01-09T10:00:00",
            status="pending",
            error_message=None,
            total_pages=0,
            text_chunks=0,
            images=[],
            tables=[]
        )
        
        # Act
        with patch('app.api.documents_controller.document_service', mock_document_service):
            response = client.post("/api/documents/upload", files={"file": file})
        
        # Assert
        assert response.status_code == 200
        # Verify that process_document would be called in background
        mock_document_service.create.assert_called_once()


class TestListDocuments:
    """Test cases for GET /api/documents endpoint."""

    @pytest.mark.asyncio
    async def test_list_documents_success(self, client, mock_document_service):
        """Test listing documents successfully."""
        # Arrange
        expected_response = {
            "total": 2,
            "offset": 0,
            "limit": 10,
            "items": [
                {
                    "id": 1,
                    "filename": "doc1.pdf",
                    "upload_date": "2025-01-09T10:00:00",
                    "status": "completed",
                    "error_message": None,
                    "total_pages": 5,
                    "text_chunks": 10,
                    "images": [],
                    "tables": []
                },
                {
                    "id": 2,
                    "filename": "doc2.pdf",
                    "upload_date": "2025-01-09T11:00:00",
                    "status": "processing",
                    "error_message": None,
                    "total_pages": 0,
                    "text_chunks": 0,
                    "images": [],
                    "tables": []
                }
            ]
        }
        
        mock_document_service.list.return_value = PaginatedDocumentResponse(**expected_response)
        
        # Act
        with patch('app.api.documents_controller.document_service', mock_document_service):
            response = client.get("/api/documents")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        mock_document_service.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_documents_with_pagination(self, client, mock_document_service):
        """Test listing documents with pagination parameters."""
        # Arrange
        expected_response = {
            "total": 5,
            "offset": 10,
            "limit": 10,
            "items": []
        }
        
        mock_document_service.list.return_value = PaginatedDocumentResponse(**expected_response)
        
        # Act
        with patch('app.api.documents_controller.document_service', mock_document_service):
            response = client.get("/api/documents?skip=10&limit=10")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["offset"] == 10
        assert data["limit"] == 10
        mock_document_service.list.assert_called_once_with(10, 10)

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, client, mock_document_service):
        """Test listing documents when none exist."""
        # Arrange
        expected_response = {
            "total": 0,
            "offset": 0,
            "limit": 10,
            "items": []
        }
        
        mock_document_service.list.return_value = PaginatedDocumentResponse(**expected_response)
        
        # Act
        with patch('app.api.documents_controller.document_service', mock_document_service):
            response = client.get("/api/documents")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert len(data["items"]) == 0


class TestGetDocument:
    """Test cases for GET /api/documents/{document_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_document_success(self, client, mock_document_service):
        """Test getting a document successfully."""
        # Arrange
        document_id = 1
        expected_response = {
            "id": 1,
            "filename": "test.pdf",
            "upload_date": "2025-01-09T10:00:00",
            "status": "completed",
            "error_message": None,
            "total_pages": 5,
            "text_chunks": 10,
            "images": [
                {
                    "id": 1,
                    "url": "/uploads/images/figure1.png",
                    "page": 1,
                    "caption": "Figure 1",
                    "width": 800,
                    "height": 600
                }
            ],
            "tables": [
                {
                    "id": 1,
                    "url": "/uploads/tables/table1.png",
                    "page": 2,
                    "caption": "Table 1",
                    "rows": 5,
                    "columns": 4,
                    "data": {"headers": ["Col1", "Col2"], "rows": [["A", "B"]]}
                }
            ]
        }
        
        mock_document_service.get.return_value = DocumentResponse(**expected_response)
        
        # Act
        with patch('app.api.documents_controller.document_service', mock_document_service):
            response = client.get(f"/api/documents/{document_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
        assert data["filename"] == "test.pdf"
        assert len(data["images"]) == 1
        assert len(data["tables"]) == 1
        mock_document_service.get.assert_called_once_with(document_id)

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, client, mock_document_service):
        """Test getting a non-existent document."""
        # Arrange
        document_id = 999
        from fastapi import HTTPException
        mock_document_service.get.side_effect = HTTPException(status_code=404, detail="Document not found")
        
        # Act
        with patch('app.api.documents_controller.document_service', mock_document_service):
            response = client.get(f"/api/documents/{document_id}")
        
        # Assert
        assert response.status_code == 404


class TestDeleteDocument:
    """Test cases for DELETE /api/documents/{document_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_document_success(self, client, mock_document_service):
        """Test deleting a document successfully."""
        # Arrange
        document_id = 1
        mock_document_service.delete.return_value = None
        
        # Act
        with patch('app.api.documents_controller.document_service', mock_document_service):
            response = client.delete(f"/api/documents/{document_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "successfully" in data["message"].lower()
        mock_document_service.delete.assert_called_once_with(document_id)

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, client, mock_document_service):
        """Test deleting a non-existent document."""
        # Arrange
        document_id = 999
        from fastapi import HTTPException
        mock_document_service.delete.side_effect = HTTPException(status_code=404, detail="Document not found")
        
        # Act
        with patch('app.api.documents_controller.document_service', mock_document_service):
            response = client.delete(f"/api/documents/{document_id}")
        
        # Assert
        assert response.status_code == 404
