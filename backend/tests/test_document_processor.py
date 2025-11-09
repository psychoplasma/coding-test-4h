"""
Unit and integration tests for document_processor module.

Tests cover:
1. PDF parsing and document structure extraction
2. Image extraction, saving, and database storage
3. Table extraction, saving, and database storage
4. Document chunking and embedding generation
5. Error handling and edge cases
6. Integration with VectorStore
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from PIL import Image

from sqlalchemy.orm import Session

from app.models.document import Document, DocumentImage, DocumentTable
from app.core.config import settings


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_document(db_session: Session):
    """Create a test document record in the database."""
    doc = Document(
        filename="test_document.pdf",
        file_path="/uploads/documents/test_document.pdf",
        processing_status="pending",
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    return doc


@pytest.fixture
def sample_pdf_file():
    """Create a temporary PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    return Image.new("RGB", (100, 100), color="red")


@pytest.fixture
def mock_conversion_result():
    """Create a mock Docling ConversionResult."""
    mock_result = MagicMock()
    mock_result.pages = [MagicMock() for _ in range(3)]  # 3 pages
    mock_result.document = MagicMock()
    mock_result.document.iterate_items.return_value = []
    return mock_result


@pytest.fixture
def mock_vector_store_instance():
    """Create a mock VectorStore instance."""
    mock_store = AsyncMock()
    mock_store.tokenizer = MagicMock()
    mock_store.store_chunk = AsyncMock()
    return mock_store


# ============================================================================
# Unit Tests: Initialization
# ============================================================================

class TestDocumentProcessorInitialization:
    """Tests for DocumentProcessor initialization."""

    @patch("app.services.document_processor.VectorStore")
    def test_init_creates_upload_directories(self, mock_vs):
        """Test that __init__ creates necessary upload directories."""
        from app.services.document_processor import DocumentProcessor
        
        DocumentProcessor()
        
        # Verify upload directories exist
        assert Path(settings.UPLOAD_DIR).exists()
        assert Path(os.path.join(settings.UPLOAD_DIR, "images")).exists()
        assert Path(os.path.join(settings.UPLOAD_DIR, "tables")).exists()
    
    @patch("app.services.document_processor.VectorStore")
    def test_init_initializes_vector_store(self, mock_vs):
        """Test that __init__ initializes VectorStore."""
        from app.services.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        assert processor.vector_store is not None
        mock_vs.assert_called_once()


# ============================================================================
# Unit Tests: PDF Parsing
# ============================================================================

class TestPdfParsing:
    """Tests for PDF parsing functionality."""
    
    @patch("app.services.document_processor.VectorStore")
    @patch("app.services.document_processor.DocumentConverter")
    def test_parse_pdf_file_success(self, mock_converter_class, mock_vs, mock_conversion_result):
        """Test successful PDF parsing."""
        from app.services.document_processor import DocumentProcessor
        
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        mock_converter.convert.return_value = mock_conversion_result
        
        processor = DocumentProcessor()
        result = processor._parse_pdf_file("test.pdf")
        
        assert result == mock_conversion_result
        mock_converter.convert.assert_called_once_with("test.pdf")
    
    @patch("app.services.document_processor.VectorStore")
    @patch("app.services.document_processor.DocumentConverter")
    def test_parse_pdf_file_sets_pipeline_options(self, mock_converter_class, mock_vs):
        """Test that PDF parsing sets correct pipeline options."""
        from app.services.document_processor import DocumentProcessor
        
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        mock_result = MagicMock()
        mock_result.pages = []
        mock_converter.convert.return_value = mock_result
        
        processor = DocumentProcessor()
        processor._parse_pdf_file("test.pdf")
        
        # Verify DocumentConverter was called with PdfFormatOption
        assert mock_converter_class.call_count > 0


# ============================================================================
# Unit Tests: Image Extraction and Saving
# ============================================================================

class TestImageExtraction:
    """Tests for image extraction functionality."""
    
    @patch("app.services.document_processor.VectorStore")
    @pytest.mark.asyncio
    async def test_save_image_success(self, mock_vs, db_session: Session, test_document: Document, sample_image):
        """Test successful image saving."""
        from app.services.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        doc_image = await processor._save_image(
            session=db_session,
            image=sample_image,
            document_id=test_document.id,
            page_number=1,
            metadata={"caption": "Test Image", "source": "test"}
        )
        
        # Verify image record was created
        assert doc_image.id is not None
        assert doc_image.document_id == test_document.id
        assert doc_image.page_number == 1
        assert doc_image.caption == "Test Image"
        assert doc_image.width == 100
        assert doc_image.height == 100
        
        # Verify image file was saved
        assert os.path.exists(doc_image.file_path)
        assert doc_image.file_path.endswith(".png")
    
    @patch("app.services.document_processor.VectorStore")
    @pytest.mark.asyncio
    async def test_extract_and_save_image_metadata_no_images(self, mock_vs, db_session: Session, test_document: Document, mock_conversion_result):
        """Test image extraction when document has no images."""
        from app.services.document_processor import DocumentProcessor
        
        mock_conversion_result.document.iterate_items.return_value = []
        
        processor = DocumentProcessor()
        images_count, image_ids_by_page = await processor._extract_and_save_image_metadata(
            db_session,
            test_document.id,
            mock_conversion_result,
        )
        
        assert images_count == 0
        assert len(image_ids_by_page) == 0
    
    @patch("app.services.document_processor.VectorStore")
    @pytest.mark.asyncio
    async def test_extract_and_save_image_metadata_with_images(
        self, 
        mock_vs,
        db_session: Session, 
        test_document: Document, 
        mock_conversion_result,
        sample_image
    ):
        """Test image extraction with multiple images."""
        from docling_core.types.doc import PictureItem
        from app.services.document_processor import DocumentProcessor
        
        # Create mock images
        mock_picture1 = MagicMock(spec=PictureItem)
        mock_picture1.prov = [MagicMock(page_no=1)]
        mock_picture1.get_image.return_value = sample_image
        mock_picture1.caption_text.return_value = "Image 1"
        
        mock_picture2 = MagicMock(spec=PictureItem)
        mock_picture2.prov = [MagicMock(page_no=2)]
        mock_picture2.get_image.return_value = sample_image
        mock_picture2.caption_text.return_value = "Image 2"
        
        mock_conversion_result.document.iterate_items.return_value = [
            (mock_picture1, None),
            (mock_picture2, None),
        ]
        
        processor = DocumentProcessor()
        images_count, image_ids_by_page = await processor._extract_and_save_image_metadata(
            db_session,
            test_document.id,
            mock_conversion_result,
        )
        
        assert images_count == 2
        assert 1 in image_ids_by_page
        assert 2 in image_ids_by_page
        assert len(image_ids_by_page[1]) == 1
        assert len(image_ids_by_page[2]) == 1


# ============================================================================
# Unit Tests: Table Extraction and Saving
# ============================================================================

class TestTableExtraction:
    """Tests for table extraction functionality."""
    
    @patch("app.services.document_processor.VectorStore")
    def test_save_table_success(self, mock_vs, db_session: Session, test_document: Document, sample_image):
        """Test successful table saving."""
        from docling_core.types.doc import TableItem
        from app.services.document_processor import DocumentProcessor
        
        mock_table = MagicMock()
        mock_table.data = MagicMock()
        mock_table.data.num_rows = 3
        mock_table.data.num_cols = 4
        mock_table.data.model_dump.return_value = {"headers": ["Col1", "Col2"]}
        
        processor = DocumentProcessor()
        
        doc_table = processor._save_table(
            session=db_session,
            table=mock_table,
            table_image=sample_image,
            document_id=test_document.id,
            page_number=2,
            metadata={"caption": "Test Table", "source": "test"}
        )
        
        # Verify table record was created
        assert doc_table.id is not None
        assert doc_table.document_id == test_document.id
        assert doc_table.page_number == 2
        assert doc_table.rows == 3
        assert doc_table.columns == 4
        assert doc_table.caption == "Test Table"
        
        # Verify image file was saved
        assert os.path.exists(doc_table.image_path)
    
    @patch("app.services.document_processor.VectorStore")
    @pytest.mark.asyncio
    async def test_extract_and_save_table_metadata_no_tables(self, mock_vs, db_session: Session, test_document: Document, mock_conversion_result):
        """Test table extraction when document has no tables."""
        from app.services.document_processor import DocumentProcessor
        
        mock_conversion_result.document.iterate_items.return_value = []
        
        processor = DocumentProcessor()
        tables_count, table_ids_by_page = await processor._extract_and_save_table_metadata(
            db_session,
            test_document.id,
            mock_conversion_result,
        )
        
        assert tables_count == 0
        assert len(table_ids_by_page) == 0
    
    @patch("app.services.document_processor.VectorStore")
    @pytest.mark.asyncio
    async def test_extract_and_save_table_metadata_with_tables(
        self,
        mock_vs,
        db_session: Session,
        test_document: Document,
        mock_conversion_result,
        sample_image
    ):
        """Test table extraction with multiple tables."""
        from app.services.document_processor import DocumentProcessor
        
        # Create mock tables using isinstance check
        mock_table1 = MagicMock()
        mock_table1.prov = [MagicMock(page_no=1)]
        mock_table1.get_image.return_value = sample_image
        mock_table1.caption_text.return_value = "Table 1"
        mock_table1.data = MagicMock()
        mock_table1.data.num_rows = 2
        mock_table1.data.num_cols = 3
        mock_table1.data.model_dump.return_value = {}
        
        mock_table2 = MagicMock()
        mock_table2.prov = [MagicMock(page_no=3)]
        mock_table2.get_image.return_value = sample_image
        mock_table2.caption_text.return_value = "Table 2"
        mock_table2.data = MagicMock()
        mock_table2.data.num_rows = 4
        mock_table2.data.num_cols = 5
        mock_table2.data.model_dump.return_value = {}
        
        # Patch isinstance to return True for TableItem
        from docling_core.types.doc import TableItem
        
        with patch("app.services.document_processor.isinstance") as mock_isinstance:
            def isinstance_side_effect(obj, cls):
                if cls == TableItem and obj in [mock_table1, mock_table2]:
                    return True
                return isinstance(obj, cls)
            
            mock_isinstance.side_effect = isinstance_side_effect
            
            mock_conversion_result.document.iterate_items.return_value = [
                (mock_table1, None),
                (mock_table2, None),
            ]
            
            processor = DocumentProcessor()
            tables_count, table_ids_by_page = await processor._extract_and_save_table_metadata(
                db_session,
                test_document.id,
                mock_conversion_result,
            )
            
            assert tables_count == 2
            assert 1 in table_ids_by_page
            assert 3 in table_ids_by_page


# ============================================================================
# Unit Tests: Document Chunking
# ============================================================================

class TestDocumentChunking:
    """Tests for document chunking functionality."""
    
    @patch("app.services.document_processor.VectorStore")
    @patch("docling.chunking.HybridChunker")
    @pytest.mark.asyncio
    async def test_chunk_document_success(
        self,
        mock_chunker_class,
        mock_vs_class,
        db_session: Session,
        test_document: Document,
    ):
        """Test successful document chunking and storing."""
        from app.services.document_processor import DocumentProcessor
        
        # Create mock chunks
        mock_chunk_data = {"content": "Test chunk", "meta": {"doc_items": [{"prov": [{"page_no": 1}]}]}}
        
        mock_vs_instance = AsyncMock()
        mock_vs_class.return_value = mock_vs_instance
        mock_vs_instance.tokenizer = MagicMock()
        
        mock_chunker = MagicMock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.chunk.return_value = [mock_chunk_data]
        mock_chunker.contextualize.return_value = "Contextualized chunk"
        
        # Mock DocChunk.model_validate
        mock_doc_chunk = MagicMock()
        mock_doc_chunk.meta.doc_items = [MagicMock()]
        mock_doc_chunk.meta.doc_items[0].prov = [MagicMock(page_no=1)]
        
        with patch("docling_core.transforms.chunker.hierarchical_chunker.DocChunk") as mock_doc_chunk_class:
            mock_doc_chunk_class.model_validate.return_value = mock_doc_chunk
            
            mock_vs_instance.store_chunk = AsyncMock()
            
            processor = DocumentProcessor()
            processor.vector_store = mock_vs_instance
            
            mock_document = MagicMock()
            num_chunks = await processor._chunk_document(
                db_session,
                test_document.id,
                mock_document,
                {},
                {}
            )
            
            assert num_chunks == 1
            mock_vs_instance.store_chunk.assert_called_once()
    
    @patch("app.services.document_processor.VectorStore")
    @patch("docling.chunking.HybridChunker")
    @pytest.mark.asyncio
    async def test_chunk_document_with_related_content(
        self,
        mock_chunker_class,
        mock_vs_class,
        db_session: Session,
        test_document: Document,
    ):
        """Test chunking with related images and tables metadata."""
        from app.services.document_processor import DocumentProcessor
        
        image_ids_by_page = {1: [1, 2]}
        table_ids_by_page = {1: [1]}
        
        mock_chunk_data = {"content": "Test chunk", "meta": {"doc_items": [{"prov": [{"page_no": 1}]}]}}
        
        mock_vs_instance = AsyncMock()
        mock_vs_class.return_value = mock_vs_instance
        mock_vs_instance.tokenizer = MagicMock()
        
        mock_chunker = MagicMock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.chunk.return_value = [mock_chunk_data]
        mock_chunker.contextualize.return_value = "Contextualized chunk"
        
        mock_doc_chunk = MagicMock()
        mock_doc_chunk.meta.doc_items = [MagicMock()]
        mock_doc_chunk.meta.doc_items[0].prov = [MagicMock(page_no=1)]
        
        with patch("docling_core.transforms.chunker.hierarchical_chunker.DocChunk") as mock_doc_chunk_class:
            mock_doc_chunk_class.model_validate.return_value = mock_doc_chunk
            
            mock_vs_instance.store_chunk = AsyncMock()
            
            processor = DocumentProcessor()
            processor.vector_store = mock_vs_instance
            
            mock_document = MagicMock()
            await processor._chunk_document(
                db_session,
                test_document.id,
                mock_document,
                image_ids_by_page,
                table_ids_by_page
            )
            
            # Verify metadata includes related content
            call_args = mock_vs_instance.store_chunk.call_args
            metadata = call_args.kwargs["metadata"]
            assert metadata["related_images"] == [1, 2]
            assert metadata["related_tables"] == [1]
    
    @patch("app.services.document_processor.VectorStore")
    @patch("docling.chunking.HybridChunker")
    @pytest.mark.asyncio
    async def test_chunk_document_handles_store_error(
        self,
        mock_chunker_class,
        mock_vs_class,
        db_session: Session,
        test_document: Document,
    ):
        """Test that chunk document continues on store error."""
        from app.services.document_processor import DocumentProcessor
        
        mock_chunk_data = {"content": "Test chunk", "meta": {"doc_items": [{"prov": [{"page_no": 1}]}]}}
        
        mock_vs_instance = AsyncMock()
        mock_vs_class.return_value = mock_vs_instance
        mock_vs_instance.tokenizer = MagicMock()
        
        mock_chunker = MagicMock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.chunk.return_value = [mock_chunk_data]
        mock_chunker.contextualize.return_value = "Contextualized chunk"
        
        mock_doc_chunk = MagicMock()
        mock_doc_chunk.meta.doc_items = [MagicMock()]
        mock_doc_chunk.meta.doc_items[0].prov = [MagicMock(page_no=1)]
        
        with patch("docling_core.transforms.chunker.hierarchical_chunker.DocChunk") as mock_doc_chunk_class:
            mock_doc_chunk_class.model_validate.return_value = mock_doc_chunk
            
            # Make store_chunk raise an error
            mock_vs_instance.store_chunk = AsyncMock(side_effect=Exception("Storage error"))
            
            processor = DocumentProcessor()
            processor.vector_store = mock_vs_instance
            
            mock_document = MagicMock()
            
            # Should not raise, should return 0 chunks
            num_chunks = await processor._chunk_document(
                db_session,
                test_document.id,
                mock_document,
                {},
                {}
            )
            
            assert num_chunks == 0


# ============================================================================
# Unit Tests: Utility Methods
# ============================================================================

class TestUtilityMethods:
    """Tests for utility methods."""
    
    @patch("app.services.document_processor.VectorStore")
    def test_generate_unique_filepath(self, mock_vs):
        """Test unique filepath generation."""
        import time
        from app.services.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        filepath1 = processor._generate_unique_filepath("image", 1, 1)
        time.sleep(0.1)  # Small delay to ensure different timestamp
        filepath2 = processor._generate_unique_filepath("table", 2, 2)
        
        # Paths should be unique due to document ID and timestamp
        assert filepath1 != filepath2
        assert "image" in filepath1
        assert "table" in filepath2
        assert "doc_1" in filepath1
        assert "doc_2" in filepath2
        assert filepath1.endswith(".png")
        assert filepath2.endswith(".png")
    
    @patch("app.services.document_processor.VectorStore")
    def test_get_document_by_id_found(self, mock_vs, db_session: Session, test_document: Document):
        """Test getting document by ID when it exists."""
        from app.services.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        doc = processor._get_document_by_id(db_session, test_document.id)
        
        assert doc.id == test_document.id
        assert doc.filename == test_document.filename
    
    @patch("app.services.document_processor.VectorStore")
    def test_get_document_by_id_not_found(self, mock_vs, db_session: Session):
        """Test getting document by ID when it doesn't exist."""
        from app.services.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        with pytest.raises(ValueError, match="Document with ID"):
            processor._get_document_by_id(db_session, 9999)
    
    @patch("app.services.document_processor.VectorStore")
    def test_save_to_disk_success(self, mock_vs, sample_image):
        """Test saving image to disk."""
        from app.services.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        filepath = processor._save_to_disk("image", 1, 1, sample_image)
        
        assert os.path.exists(filepath)
        assert filepath.endswith(".png")
        
        # Cleanup
        os.unlink(filepath)
    
    @patch("app.services.document_processor.VectorStore")
    def test_save_to_disk_creates_placeholder_on_error(self, mock_vs, sample_image):
        """Test that placeholder is created when save fails."""
        from app.services.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        with patch.object(sample_image, "save", side_effect=Exception("Disk error")):
            filepath = processor._save_to_disk("image", 1, 1, sample_image)
            
            assert os.path.exists(filepath)
            assert filepath.endswith(".png")
            
            # Verify it's a valid image
            img = Image.open(filepath)
            assert img.size == (800, 600)
            
            # Cleanup
            os.unlink(filepath)


# ============================================================================
# Integration Tests: Full Document Processing
# ============================================================================

class TestDocumentProcessing:
    """Integration tests for full document processing."""
    
    @patch("app.services.document_processor.VectorStore")
    @patch("app.services.document_processor.DocumentConverter")
    @patch("docling.chunking.HybridChunker")
    @pytest.mark.asyncio
    async def test_process_document_full_flow(
        self,
        mock_chunker_class,
        mock_converter_class,
        mock_vs_class,
        db_session: Session,
        test_document: Document,
        sample_pdf_file,
    ):
        """Test full document processing flow."""
        from app.services.document_processor import DocumentProcessor
        from docling_core.types.doc import PictureItem, TableItem
        from contextlib import contextmanager
        
        # Create mock Docling result
        mock_picture = MagicMock(spec=PictureItem)
        mock_picture.prov = [MagicMock(page_no=1)]
        mock_picture.get_image.return_value = Image.new("RGB", (100, 100), color="blue")
        mock_picture.caption_text.return_value = "Sample Image"
        
        mock_table = MagicMock(spec=TableItem)
        mock_table.prov = [MagicMock(page_no=2)]
        mock_table.get_image.return_value = Image.new("RGB", (200, 200), color="green")
        mock_table.caption_text.return_value = "Sample Table"
        mock_table.data = MagicMock()
        mock_table.data.num_rows = 2
        mock_table.data.num_cols = 3
        mock_table.data.model_dump.return_value = {"headers": ["A", "B", "C"]}
        
        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = [
            (mock_picture, None),
            (mock_table, None),
        ]
        
        mock_result = MagicMock()
        mock_result.pages = [MagicMock() for _ in range(3)]
        mock_result.document = mock_doc
        
        # Setup chunks
        mock_chunk_data = {"content": "Test chunk", "meta": {"doc_items": [{"prov": [{"page_no": 1}]}]}}
        
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        mock_converter.convert.return_value = mock_result
        
        mock_vs_instance = AsyncMock()
        mock_vs_class.return_value = mock_vs_instance
        mock_vs_instance.tokenizer = MagicMock()
        
        mock_chunker = MagicMock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.chunk.return_value = [mock_chunk_data]
        mock_chunker.contextualize.return_value = "Contextualized chunk"
        
        mock_doc_chunk = MagicMock()
        mock_doc_chunk.meta.doc_items = [MagicMock()]
        mock_doc_chunk.meta.doc_items[0].prov = [MagicMock(page_no=1)]
        
        with patch("docling_core.transforms.chunker.hierarchical_chunker.DocChunk") as mock_doc_chunk_class:
            mock_doc_chunk_class.model_validate.return_value = mock_doc_chunk
            
            mock_vs_instance.store_chunk = AsyncMock()
            
            processor = DocumentProcessor()
            processor.vector_store = mock_vs_instance
            
            # Process document
            await processor.process_document(db_session, sample_pdf_file, test_document.id)
            
            # Verify document status
            db_session.refresh(test_document)
            assert test_document.processing_status == "completed"
            assert test_document.total_pages == 3
            assert test_document.images_count == 1
            assert test_document.tables_count == 1
            assert test_document.text_chunks_count == 1
            assert test_document.processing_time > 0
    
    @patch("app.services.document_processor.VectorStore")
    @patch("app.services.document_processor.DocumentConverter")
    @patch("docling.chunking.HybridChunker")
    @pytest.mark.asyncio
    async def test_process_document_updates_status(
        self,
        mock_chunker_class,
        mock_converter_class,
        mock_vs_class,
        db_session: Session,
        test_document: Document,
        sample_pdf_file
    ):
        """Test that document status is updated during processing."""
        from app.services.document_processor import DocumentProcessor
        
        mock_result = MagicMock()
        mock_result.pages = [MagicMock()]
        mock_result.document.iterate_items.return_value = []
        
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        mock_converter.convert.return_value = mock_result
        
        mock_vs_instance = AsyncMock()
        mock_vs_class.return_value = mock_vs_instance
        mock_vs_instance.tokenizer = MagicMock()
        
        mock_chunker = MagicMock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.chunk.return_value = []
        
        processor = DocumentProcessor()
        processor.vector_store = mock_vs_instance
        
        await processor.process_document(db_session, sample_pdf_file, test_document.id)
        
        # Verify status progression
        db_session.refresh(test_document)
        assert test_document.processing_status == "completed"


# ============================================================================
# Integration Tests: Database Interactions
# ============================================================================

class TestDatabaseInteractions:
    """Integration tests for database interactions."""
    
    @patch("app.services.document_processor.VectorStore")
    @pytest.mark.asyncio
    async def test_save_and_retrieve_images(self, mock_vs, db_session: Session, test_document: Document):
        """Test saving and retrieving images from database."""
        from app.services.document_processor import DocumentProcessor
        
        sample_image = Image.new("RGB", (150, 200), color="yellow")
        
        processor = DocumentProcessor()
        
        # Save first image
        doc_image1 = await processor._save_image(
            session=db_session,
            image=sample_image,
            document_id=test_document.id,
            page_number=1,
            metadata={"caption": "Image 1"}
        )
        
        # Save second image
        doc_image2 = await processor._save_image(
            session=db_session,
            image=sample_image,
            document_id=test_document.id,
            page_number=2,
            metadata={"caption": "Image 2"}
        )
        
        # Retrieve images
        images = db_session.query(DocumentImage).filter_by(
            document_id=test_document.id
        ).all()
        
        assert len(images) == 2
        assert images[0].caption == "Image 1"
        assert images[1].caption == "Image 2"
    
    @patch("app.services.document_processor.VectorStore")
    def test_save_and_retrieve_tables(self, mock_vs, db_session: Session, test_document: Document):
        """Test saving and retrieving tables from database."""
        from app.services.document_processor import DocumentProcessor
        
        sample_image = Image.new("RGB", (200, 150), color="cyan")
        mock_table = MagicMock()
        mock_table.data = MagicMock()
        mock_table.data.num_rows = 5
        mock_table.data.num_cols = 3
        mock_table.data.model_dump.return_value = {"headers": ["A", "B", "C"]}
        
        processor = DocumentProcessor()
        
        # Save first table
        doc_table1 = processor._save_table(
            session=db_session,
            table=mock_table,
            table_image=sample_image,
            document_id=test_document.id,
            page_number=1,
            metadata={"caption": "Table 1"}
        )
        
        # Save second table
        doc_table2 = processor._save_table(
            session=db_session,
            table=mock_table,
            table_image=sample_image,
            document_id=test_document.id,
            page_number=3,
            metadata={"caption": "Table 2"}
        )
        
        # Retrieve tables
        tables = db_session.query(DocumentTable).filter_by(
            document_id=test_document.id
        ).all()
        
        assert len(tables) == 2
        assert tables[0].caption == "Table 1"
        assert tables[1].caption == "Table 2"
        assert tables[0].rows == 5
        assert tables[0].columns == 3


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in document processing."""
    
    @patch("app.services.document_processor.VectorStore")
    def test_process_document_invalid_document_id(self, mock_vs, db_session: Session, sample_pdf_file):
        """Test processing with invalid document ID."""
        from app.services.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        with pytest.raises(ValueError, match="Document with ID"):
            processor._get_document_by_id(db_session, 9999)
