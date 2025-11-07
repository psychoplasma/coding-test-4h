"""
Vector store service using pgvector.

Implements:
1. Generate embeddings for text chunks using LangChain (OpenAI or sentence-transformers)
2. Store embeddings in PostgreSQL with pgvector
3. Perform similarity search with cosine similarity
4. Link related images and tables
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import text
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings

from app.core.config import settings
from app.db.session import scoped_session
from app.models.document import DocumentChunk, DocumentImage, DocumentTable


logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store for document embeddings and similarity search using pgvector.
    
    Supports two embedding models:
    - OpenAI (if OPENAI_API_KEY is provided)
    - Sentence Transformers (local fallback)
    """
    
    def __init__(self):
        self.embeddings_model = None
        self._initialize_embeddings_model()
        self._ensure_extension()
    
    def _initialize_embeddings_model(self):
        """
        Initialize embeddings model using LangChain.
        
        Supports:
        - OpenAI embeddings via OpenAIEmbeddings (if OPENAI_API_KEY is provided)
        - Gemini embeddings via VertexAIEmbeddings (if GEMINI_API_KEY is provided)
        - HuggingFace embeddings as local fallback
        """
        if settings.OPENAI_API_KEY:
            try:
                self.embeddings_model = OpenAIEmbeddings(
                    model=settings.OPENAI_EMBEDDING_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                    dimensions=settings.EMBEDDING_DIMENSION,
                )
                logger.info(f"Using OpenAI embeddings (model: {settings.OPENAI_EMBEDDING_MODEL})")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI embeddings: {e}. Falling back to HuggingFace embeddings")
                self._init_sentence_transformers()
        elif settings.USE_GEMINI:
            try:
                # Dimension is 768 by default, if needed, can be adjusted when calling embed(dimension=),
                # instead of embed_query which does not pass dimension parameter down to embed() method.
                # Just skipping dimension setting for the sake of simplicity
                self.embeddings_model = VertexAIEmbeddings(
                    model=settings.GEMINI_EMBEDDING_MODEL,
                    api_key=settings.GEMINI_API_KEY,
                )
                logger.info(f"Using Gemini embeddings (model: {settings.GEMINI_EMBEDDING_MODEL})")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini embeddings: {e}. Falling back to HuggingFace embeddings")
                self._init_sentence_transformers()
        else:
            self._init_sentence_transformers()
    
    def _init_sentence_transformers(self):
        """
        Initialize HuggingFace embeddings (sentence-transformers) as fallback.

        Uses the all-MiniLM-L6-v2 model with 384 dimensions for local embeddings.
        """
        try:
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            )
            logger.info("Using HuggingFace embeddings (model: all-MiniLM-L6-v2)")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embeddings: {e}")
            raise RuntimeError("Could not initialize any embedding model")

    def _ensure_extension(self):
        """
        Ensure pgvector extension is enabled in PostgreSQL.
        
        Creates the vector extension if it doesn't exist.
        """
        with scoped_session() as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()
            logger.info("pgvector extension enabled")
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        Uses OpenAI API if configured, otherwise falls back to HuggingFace embeddings.
        
        Args:
            text: Text to embed
            
        Returns:
            numpy array of embeddings (1536-dimensional for OpenAI, 384 for HuggingFace)
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        try:
            embedding = self.embeddings_model.embed_query(text)
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}")
    
    async def store_chunk(
        self,
        session: Session,
        content: str, 
        document_id: int,
        page_number: int,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentChunk:
        """
        Store a text chunk with its embedding in the database.
        
        Args:
            content: Text content of the chunk
            document_id: ID of the parent document
            page_number: Page number where chunk appears
            chunk_index: Index of chunk in document
            metadata: Additional metadata (related_images, related_tables, etc.)
            
        Returns:
            Created DocumentChunk record
        """
        # Generate embedding for the content
        embedding = await self.generate_embedding(content)
        
        # Create DocumentChunk record
        chunk = DocumentChunk(
            document_id=document_id,
            content=content,
            embedding=embedding.tolist(),  # pgvector accepts list
            page_number=page_number,
            chunk_index=chunk_index,
            chunk_metadata=metadata or {}
        )
        
        # Store in database
        session.add(chunk)
        session.flush()  # To get ID assigned

        logger.info(f"Stored chunk {chunk_index} for document {document_id}")
        return chunk
    
    async def similarity_search(
        self,
        session: Session,
        query: str,
        document_ids: Optional[List[int]] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.

        Steps:
        1. Generate embedding for query
        2. Use pgvector's cosine similarity (<=> operator)
        3. Filter by document_id if provided
        4. Return top k results with scores
        5. Include related images and tables in results
        
        Example SQL query:
        SELECT 
            id,
            content,
            page_number,
            metadata,
            1 - (embedding <=> :query_embedding) as similarity
        FROM document_chunks
        WHERE document_id = :document_id  -- optional filter
        ORDER BY embedding <=> :query_embedding
        LIMIT :k
        
        Args:
            query: Search query text
            document_id: Optional document ID to filter
            k: Number of results to return (default 5)
            
        Returns:
            List of similar chunks with metadata:
            [
                {
                    "id": 123,
                    "content": "...",
                    "score": 0.95,
                    "page_number": 3,
                    "metadata": {...},
                },
                ...
            ]
            
        Raises:
            RuntimeError: If embedding generation or search fails
        """
        # Generate embedding for query
        query_embedding = await self.generate_embedding(query)
        
        # Build SQL query with pgvector similarity search
        sql_query = """
            SELECT 
                dc.id,
                dc.content,
                dc.page_number,
                dc.chunk_index,
                dc.metadata,
                dc.document_id,
                1 - (dc.embedding <=> :query_embedding::vector) as similarity_score
            FROM document_chunks dc
        """
        
        # Add document filter if specified
        if document_ids is not None and len(document_ids) > 0:
            sql_query += " WHERE dc.document_id IN :document_ids"
        
        # Order by similarity and limit
        sql_query += """
            ORDER BY dc.embedding <=> :query_embedding::vector
            LIMIT :k
        """
        
        # Execute query
        params = {
            "query_embedding": query_embedding.tolist(),
            "k": k
        }
        if document_ids is not None and len(document_ids) > 0:
            params["document_ids"] = document_ids
        
        rows = session.execute(text(sql_query), params).all()
        
        # Format results
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "content": row[1],
                "page_number": row[2],
                "chunk_index": row[3],
                "metadata": row[4] or {},
                "document_id": row[5],
                "score": float(row[6]) if row[6] else 0.0
            })
        
        logger.info(f"Found {len(results)} similar chunks for query")
        return results

    async def get_related_content(
        self,
        session: Session,
        chunk_ids: List[int]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get related images and tables for given chunks.
        
        Retrieves related media referenced in chunk metadata.
        
        Args:
            chunk_ids: List of chunk IDs to get related content for
            
        Returns:
            Dictionary with images and tables:
            {
                "images": [
                    {
                        "id": 1,
                        "file_path": "...",
                        "page_number": 3,
                        "caption": "...",
                        "width": 800,
                        "height": 600,
                        "metadata": {...}
                    },
                    ...
                ],
                "tables": [
                    {
                        "id": 1,
                        "image_path": "...",
                        "page_number": 5,
                        "caption": "...",
                        "rows": 10,
                        "columns": 5,
                        "data": {...},
                        "metadata": {...}
                    },
                    ...
                ]
            }
            
        Raises:
            RuntimeError: If database query fails
        """
        # Extract related image and table IDs from chunks
        related_image_ids, related_table_ids = self._extract_related_media_ids_from_chunks(session, chunk_ids)

        # Retrieve related images
        images = []
        if related_image_ids:
            images = self._get_document_images(session, list(related_image_ids))

        # Retrieve related tables
        tables = []
        if related_table_ids:
            tables = self._get_document_tables(session, list(related_table_ids))

        logger.info(f"Retrieved {len(images)} images and {len(tables)} tables for {len(chunk_ids)} chunks")

        return {
            "images": images,
            "tables": tables
        }
    
    def _extract_related_media_ids_from_chunks(self, session: Session, chunk_ids: List[int]) -> Tuple[set, set]:
        """
        Extract related image and table IDs from chunk metadata.

        Args:
            chunk_ids: List of DocumentChunk IDs

        Returns:
            tuple(set, set): (related_image_ids, related_table_ids)
        """
        try:
            # Get all chunks with their metadata
            chunks = session.query(DocumentChunk).filter(
                DocumentChunk.id.in_(chunk_ids)
            ).all()

            # Collect all related image and table IDs from metadata
            related_image_ids = set()
            related_table_ids = set()

            for chunk in chunks:
                if chunk.chunk_metadata:
                    related_images = chunk.chunk_metadata.get("related_images", [])
                    related_tables = chunk.chunk_metadata.get("related_tables", [])
                    
                    if isinstance(related_images, list):
                        related_image_ids.update(related_images)
                    if isinstance(related_tables, list):
                        related_table_ids.update(related_tables)

            return related_image_ids, related_table_ids
        except Exception as e:
            logger.error(f"Error extracting related media IDs: {e}")
            raise RuntimeError(f"Failed to extract related media IDs: {e}")

    def _get_document_images(self, session: Session, image_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Retrieve all images for the given image IDs.

        Args:
            image_ids: List of image IDs

        Returns:
            List of DocumentImage records
        """        
        try:
            images = session.query(DocumentImage).filter(
                DocumentImage.id.in_(image_ids)
            ).all()

            return [
                {
                    "id": img.id,
                    "file_path": img.file_path,
                    "page_number": img.page_number,
                    "caption": img.caption,
                    "width": img.width,
                    "height": img.height,
                    "metadata": img.image_metadata or {}
                }
                for img in images
            ]
        except Exception as e:
            logger.error(f"Error retrieving document images: {e}")
            raise RuntimeError(f"Failed to retrieve document images: {e}")

    def _get_document_tables(self, session: Session, table_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Retrieve all tables for the given table IDs.

        Args:
            table_ids: List of table IDs

        Returns:
            List of DocumentTable records
        """
        try:
            tables = session.query(DocumentTable).filter(
                DocumentTable.id.in_(table_ids)
            ).all()

            return [
                {
                    "id": tbl.id,
                    "image_path": tbl.image_path,
                    "page_number": tbl.page_number,
                    "caption": tbl.caption,
                    "rows": tbl.rows,
                    "columns": tbl.columns,
                    "data": tbl.data or {},
                    "metadata": tbl.table_metadata or {}
                }
                for tbl in tables
            ]
        except Exception as e:
            logger.error(f"Error retrieving document tables: {e}")
            raise RuntimeError(f"Failed to retrieve document tables: {e}")
