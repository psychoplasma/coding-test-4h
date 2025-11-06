"""
Document management API endpoints
"""
import os
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from app.core.config import settings
from app.api.document_service import DocumentService
from app.api.dto import (
    DeleteResponse,
    DocumentResponse,
    PaginatedDocumentResponse,
    UploadDocumentResponse,
)

router = APIRouter()

document_service = DocumentService()


@router.post("/upload", response_model=UploadDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
) -> UploadDocumentResponse:
    """
    Upload a PDF document for processing
    
    This endpoint:
    1. Saves the uploaded file
    2. Creates a document record
    3. Triggers background processing (Docling extraction)
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Validate file size
    contents = await file.read()
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds {settings.MAX_FILE_SIZE / 1024 / 1024}MB limit"
        )

    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{file_id}{file_extension}"
    file_path = os.path.join(settings.UPLOAD_DIR, "documents", unique_filename)

    # Save file
    with open(file_path, "wb") as f:
        f.write(contents)

    # Create document record
    doc = await document_service.create(filename=file.filename, file_path=file_path)

    # Trigger background processing
    background_tasks.add_task(
        document_service.process_document,
        doc.id,
        file_path,
    )

    return UploadDocumentResponse(
        id=doc.id,
        filename=doc.filename,
        status=doc.processing_status,
        message="Document uploaded successfully. Processing will begin shortly."
    )

@router.get("", response_model=PaginatedDocumentResponse)
async def list_documents(skip: int = 0, limit: int = 10) -> PaginatedDocumentResponse:
    """Get list of all documents with pagination"""
    return await document_service.list(skip, limit)

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: int) -> DocumentResponse:
    """
    Get document details including extracted images and tables
    """
    return await document_service.get(document_id)

@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: int) -> DeleteResponse:
    """
    Delete a document and all associated data
    """
    await document_service.delete(document_id)
    return DeleteResponse(message="Document deleted successfully")
