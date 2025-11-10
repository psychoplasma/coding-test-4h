"""
Unit and integration tests for VectorStore class.

Tests cover:
1. VectorStore initialization with OpenAI and HuggingFace embeddings
2. Embedding generation
3. Chunk storage and retrieval
4. Similarity search with pgvector
5. Related content retrieval (images and tables)
6. Error handling and edge cases
"""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.orm import Session

from app.services.vector_store import VectorStore
from app.models.document import Document, DocumentChunk, DocumentImage, DocumentTable


# ============================================================================
# Unit Tests: Initialization
# ============================================================================

class TestVectorStoreInitialization:
    """Tests for VectorStore initialization."""
    
    @patch("app.services.vector_store.OpenAIEmbeddings")
    @patch("app.services.vector_store.scoped_session")
    def test_init_with_openai_api_key(self, mock_scoped_session, mock_openai_embeddings):
        """Test VectorStore initializes with OpenAI embeddings when API key is provided."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.core.config.settings.OPENAI_API_KEY", "test-api-key"):
            with patch("app.services.vector_store.VectorStore._initialize_openai_tokenizer"):
                store = VectorStore()
                
                assert store.embeddings_model is not None
                mock_openai_embeddings.assert_called_once()
    
    @patch("app.services.vector_store.HuggingFaceEmbeddings")
    @patch("app.services.vector_store.scoped_session")
    def test_init_with_huggingface_fallback(self, mock_scoped_session, mock_hf_embeddings):
        """Test VectorStore falls back to HuggingFace embeddings when no OpenAI key."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.core.config.settings.OPENAI_API_KEY", None):
            with patch("app.services.vector_store.VectorStore._initialize_huggingface_tokenizer"):
                store = VectorStore()
                
                assert store.embeddings_model is not None
                mock_hf_embeddings.assert_called_once()
    
    @patch("app.services.vector_store.scoped_session")
    def test_ensure_extension_creates_pgvector(self, mock_scoped_session):
        """Test that _ensure_extension creates pgvector extension."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                # Verify execute was called with CREATE EXTENSION
                assert mock_session.execute.called


# ============================================================================
# Unit Tests: Embedding Generation
# ============================================================================

class TestEmbeddingGeneration:
    """Tests for embedding generation functionality."""
    
    @pytest.mark.asyncio
    @patch("app.services.vector_store.scoped_session")
    async def test_generate_embedding_success(self, mock_scoped_session):
        """Test successful embedding generation."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                # Mock the embeddings model
                mock_embedding = [0.1] * 384  # HuggingFace sentence-transformers: 384 dims
                store.embeddings_model = MagicMock()
                store.embeddings_model.embed_query = MagicMock(return_value=mock_embedding)
                
                result = await store.generate_embedding("Test text")
                
                assert isinstance(result, np.ndarray)
                assert len(result) == 384
                assert all(x == 0.1 for x in result)
    
    @pytest.mark.asyncio
    @patch("app.services.vector_store.scoped_session")
    async def test_generate_embedding_empty_text_raises_error(self, mock_scoped_session):
        """Test that empty text raises ValueError."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                with pytest.raises(ValueError, match="Text must be a non-empty string"):
                    await store.generate_embedding("")
    
    @pytest.mark.asyncio
    @patch("app.services.vector_store.scoped_session")
    async def test_generate_embedding_invalid_type_raises_error(self, mock_scoped_session):
        """Test that non-string input raises ValueError."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                with pytest.raises(ValueError, match="Text must be a non-empty string"):
                    await store.generate_embedding(123)


# ============================================================================
# Integration Tests: Chunk Storage and Retrieval
# ============================================================================

class TestChunkStorage:
    """Tests for chunk storage functionality."""
    
    @pytest.mark.asyncio
    @patch("app.services.vector_store.scoped_session")
    async def test_store_chunk_success(self, mock_scoped_session, db_session: Session, sample_document: Document):
        """Test successful chunk storage."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                # Mock embedding generation (384 dimensions for HuggingFace)
                store.embeddings_model = MagicMock()
                store.embeddings_model.embed_query = MagicMock(return_value=[0.1] * 384)
                
                chunk = await store.store_chunk(
                    session=db_session,
                    content="This is a test chunk",
                    document_id=sample_document.id,
                    page_number=1,
                    chunk_index=0,
                    metadata={"test": "metadata"}
                )
                
                assert chunk.document_id == sample_document.id
                assert chunk.content == "This is a test chunk"
                assert chunk.page_number == 1
                assert chunk.chunk_index == 0
                assert chunk.chunk_metadata == {"test": "metadata"}
                assert len(chunk.embedding) == 384
    
    @pytest.mark.asyncio
    @patch("app.services.vector_store.scoped_session")
    async def test_store_chunk_without_metadata(self, mock_scoped_session, db_session: Session, sample_document: Document):
        """Test chunk storage without metadata."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                store.embeddings_model = MagicMock()
                store.embeddings_model.embed_query = MagicMock(return_value=[0.2] * 384)
                
                chunk = await store.store_chunk(
                    session=db_session,
                    content="Test chunk without metadata",
                    document_id=sample_document.id,
                    page_number=2,
                    chunk_index=1
                )
                
                assert chunk.chunk_metadata == {}


# ============================================================================
# Integration Tests: Similarity Search
# ============================================================================

class TestSimilaritySearch:
    """Tests for similarity search functionality."""
    
    @pytest.mark.asyncio
    @patch("app.services.vector_store.scoped_session")
    async def test_similarity_search_all_documents(
        self, 
        mock_scoped_session, 
        db_session: Session, 
        sample_chunks
    ):
        """Test similarity search across all documents."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                store.embeddings_model = MagicMock()
                store.embeddings_model.embed_query = MagicMock(return_value=[0.1] * 384)
                
                results = await store.similarity_search(
                    session=db_session,
                    query="machine learning",
                    k=3
                )
                
                assert isinstance(results, list)
                assert len(results) <= 3
                for result in results:
                    assert "id" in result
                    assert "content" in result
                    assert "score" in result
                    assert "page_number" in result
                    assert "document_id" in result
    
    @pytest.mark.asyncio
    @patch("app.services.vector_store.scoped_session")
    async def test_similarity_search_specific_documents(
        self,
        mock_scoped_session,
        db_session: Session,
        sample_document: Document,
        sample_chunks
    ):
        """Test similarity search filtered by document IDs."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                store.embeddings_model = MagicMock()
                store.embeddings_model.embed_query = MagicMock(return_value=[0.1] * 384)
                
                results = await store.similarity_search(
                    session=db_session,
                    query="neural networks",
                    document_ids=[sample_document.id],
                    k=5
                )
                
                assert isinstance(results, list)
                # All results should be from the specified document
                for result in results:
                    assert result["document_id"] == sample_document.id
    
    @pytest.mark.asyncio
    @patch("app.services.vector_store.scoped_session")
    async def test_similarity_search_returns_formatted_results(
        self,
        mock_scoped_session,
        db_session: Session,
        sample_chunks
    ):
        """Test that similarity search returns properly formatted results."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                store.embeddings_model = MagicMock()
                store.embeddings_model.embed_query = MagicMock(return_value=[0.1] * 384)
                
                results = await store.similarity_search(
                    session=db_session,
                    query="deep learning",
                    k=1
                )
                
                if results:
                    result = results[0]
                    assert "id" in result
                    assert "content" in result
                    assert "page_number" in result
                    assert "chunk_index" in result
                    assert "chunk_metadata" in result
                    assert "document_id" in result
                    assert "score" in result
                    assert isinstance(result["score"], float)
                    assert 0 <= result["score"] <= 1


# ============================================================================
# Integration Tests: Related Content Retrieval
# ============================================================================

class TestRelatedContent:
    """Tests for related content retrieval."""
    
    @pytest.mark.asyncio
    @patch("app.services.vector_store.scoped_session")
    async def test_get_related_content_with_images_and_tables(
        self,
        mock_scoped_session,
        db_session: Session,
        sample_document: Document,
        sample_chunks,
        sample_images,
        sample_tables
    ):
        """Test retrieval of related images and tables."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                # Get chunk IDs
                chunk_ids = [chunk.id for chunk in sample_chunks]
                
                content = await store.get_related_content(
                    session=db_session,
                    chunk_ids=chunk_ids
                )
                
                assert "images" in content
                assert "tables" in content
                assert isinstance(content["images"], list)
                assert isinstance(content["tables"], list)
    
    @pytest.mark.asyncio
    @patch("app.services.vector_store.scoped_session")
    async def test_get_related_content_image_format(
        self,
        mock_scoped_session,
        db_session: Session,
        sample_document: Document,
        sample_chunks,
        sample_images
    ):
        """Test that related images have correct format."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                chunk_ids = [sample_chunks[0].id]
                
                content = await store.get_related_content(
                    session=db_session,
                    chunk_ids=chunk_ids
                )
                
                for image in content["images"]:
                    assert "id" in image
                    assert "file_path" in image
                    assert "page_number" in image
                    assert "caption" in image
                    assert "width" in image
                    assert "height" in image
                    assert "metadata" in image
    
    @pytest.mark.asyncio
    @patch("app.services.vector_store.scoped_session")
    async def test_get_related_content_table_format(
        self,
        mock_scoped_session,
        db_session: Session,
        sample_document: Document,
        sample_chunks,
        sample_tables
    ):
        """Test that related tables have correct format."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                chunk_ids = [sample_chunks[0].id]
                
                content = await store.get_related_content(
                    session=db_session,
                    chunk_ids=chunk_ids
                )
                
                for table in content["tables"]:
                    assert "id" in table
                    assert "image_path" in table
                    assert "page_number" in table
                    assert "caption" in table
                    assert "rows" in table
                    assert "columns" in table
                    assert "data" in table
                    assert "metadata" in table


# ============================================================================
# Integration Tests: Media ID Extraction
# ============================================================================

class TestMediaIdExtraction:
    """Tests for extracting related media IDs from chunks."""
    
    @patch("app.services.vector_store.scoped_session")
    def test_extract_related_media_ids_from_chunks_success(
        self,
        mock_scoped_session,
        db_session: Session,
        sample_chunks
    ):
        """Test successful extraction of related media IDs."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                chunk_ids = [sample_chunks[0].id, sample_chunks[1].id]
                
                image_ids, table_ids = store._extract_related_media_ids_from_chunks(
                    session=db_session,
                    chunk_ids=chunk_ids
                )
                
                assert isinstance(image_ids, set)
                assert isinstance(table_ids, set)
    
    @patch("app.services.vector_store.scoped_session")
    def test_extract_related_media_ids_empty_chunks(
        self,
        mock_scoped_session,
        db_session: Session
    ):
        """Test extraction with empty chunk list."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                image_ids, table_ids = store._extract_related_media_ids_from_chunks(
                    session=db_session,
                    chunk_ids=[]
                )
                
                assert len(image_ids) == 0
                assert len(table_ids) == 0


# ============================================================================
# Integration Tests: Document Images Retrieval
# ============================================================================

class TestDocumentImagesRetrieval:
    """Tests for retrieving document images."""
    
    @patch("app.services.vector_store.scoped_session")
    def test_get_document_images_success(
        self,
        mock_scoped_session,
        db_session: Session,
        sample_images
    ):
        """Test successful retrieval of document images."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                image_ids = [img.id for img in sample_images]
                
                result = store._get_document_images(
                    session=db_session,
                    image_ids=image_ids
                )
                
                assert len(result) == len(sample_images)
                for img in result:
                    assert "id" in img
                    assert "file_path" in img
                    assert "page_number" in img
                    assert "caption" in img
                    assert "width" in img
                    assert "height" in img
                    assert "metadata" in img
    
    @patch("app.services.vector_store.scoped_session")
    def test_get_document_images_empty_list(
        self,
        mock_scoped_session,
        db_session: Session
    ):
        """Test retrieval with empty image list."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                result = store._get_document_images(
                    session=db_session,
                    image_ids=[]
                )
                
                assert len(result) == 0


# ============================================================================
# Integration Tests: Document Tables Retrieval
# ============================================================================

class TestDocumentTablesRetrieval:
    """Tests for retrieving document tables."""
    
    @patch("app.services.vector_store.scoped_session")
    def test_get_document_tables_success(
        self,
        mock_scoped_session,
        db_session: Session,
        sample_tables
    ):
        """Test successful retrieval of document tables."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                table_ids = [tbl.id for tbl in sample_tables]
                
                result = store._get_document_tables(
                    session=db_session,
                    table_ids=table_ids
                )
                
                assert len(result) == len(sample_tables)
                for tbl in result:
                    assert "id" in tbl
                    assert "image_path" in tbl
                    assert "page_number" in tbl
                    assert "caption" in tbl
                    assert "rows" in tbl
                    assert "columns" in tbl
                    assert "data" in tbl
                    assert "metadata" in tbl
    
    @patch("app.services.vector_store.scoped_session")
    def test_get_document_tables_empty_list(
        self,
        mock_scoped_session,
        db_session: Session
    ):
        """Test retrieval with empty table list."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                result = store._get_document_tables(
                    session=db_session,
                    table_ids=[]
                )
                
                assert len(result) == 0


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in VectorStore."""
    
    @pytest.mark.asyncio
    @patch("app.services.vector_store.scoped_session")
    async def test_generate_embedding_api_error(self, mock_scoped_session):
        """Test handling of embedding generation API errors."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                store.embeddings_model = MagicMock()
                store.embeddings_model.embed_query = MagicMock(
                    side_effect=Exception("API Error")
                )
                
                with pytest.raises(RuntimeError, match="Failed to generate embedding"):
                    await store.generate_embedding("Test text")
    
    @patch("app.services.vector_store.scoped_session")
    def test_get_document_images_db_error(self, mock_scoped_session, db_session: Session):
        """Test handling of database errors in image retrieval."""
        mock_session = MagicMock()
        mock_scoped_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scoped_session.return_value.__exit__ = MagicMock(return_value=False)
        
        with patch("app.services.vector_store.VectorStore._initialize_embeddings_model"):
            with patch("app.services.vector_store.VectorStore._initialize_tokenizer"):
                store = VectorStore()
                
                # Mock the query to raise an exception
                with patch.object(db_session, "query", side_effect=Exception("DB Error")):
                    with pytest.raises(RuntimeError, match="Failed to retrieve document images"):
                        store._get_document_images(db_session, [1, 2, 3])
