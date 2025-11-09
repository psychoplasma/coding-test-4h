"""
Test configuration and fixtures for the application.

This module provides:
1. PostgreSQL testcontainer setup with pgvector extension
2. Database session fixtures
3. Mock fixtures for LLM client and VectorStore
4. Common test data factories
"""
import asyncio
import pytest
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from testcontainers.postgres import PostgresContainer

from app.models.document import Document, DocumentChunk, DocumentImage, DocumentTable


# ============================================================================
# PostgreSQL Testcontainer Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def postgres_container() -> Generator:
    """
    Create and manage a PostgreSQL container for the entire test session.
    The container automatically sets up pgvector extension.
    """
    container = PostgresContainer(
        image="pgvector/pgvector:pg17",
        username="testuser",
        password="testpass",
        dbname="testdb"
    )
    container.start()

    # Connect and create pgvector extension
    conn_string = container.get_connection_url()
    engine = create_engine(conn_string)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    engine.dispose()

    yield container

    container.stop()


@pytest.fixture(scope="session")
def test_database_url(postgres_container: PostgresContainer) -> str:
    """
    Get the connection URL for the test PostgreSQL container.
    """
    return postgres_container.get_connection_url()


@pytest.fixture(scope="function")
def db_engine(test_database_url: str):
    """
    Create a SQLAlchemy engine for the test database.
    Uses the testcontainer PostgreSQL database.
    """
    from app.db.session import Base
    
    # Patch settings to use test database
    from app.core import config as config_module
    original_url = config_module.settings.DATABASE_URL
    config_module.settings.DATABASE_URL = test_database_url
    
    engine = create_engine(
        test_database_url,
        echo=False,
        pool_pre_ping=True,
    )
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Clean up
    Base.metadata.drop_all(engine)
    engine.dispose()
    
    # Restore original settings
    config_module.settings.DATABASE_URL = original_url


@pytest.fixture(scope="function")
def db_session(db_engine) -> Generator[Session, None, None]:
    """
    Create a database session for testing.
    Provides a new session for each test, automatically rolled back after test.
    """
    SessionLocal_override = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = SessionLocal_override()

    yield session

    session.rollback()
    session.close()


# ============================================================================
# LLM Client Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """
    Create a mock LLM client for testing.

    Returns:
        Mock ChatOpenAI/ChatVertexAI client
    """
    mock_client = MagicMock()

    # Mock the invoke method to return a response
    mock_response = MagicMock()
    mock_response.content = "This is a mock response from the LLM."
    mock_client.invoke.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_llm_response() -> str:
    """
    Provide a consistent mock LLM response.

    Returns:
        Mock response string
    """
    return "Based on the provided context, here is my answer: The document discusses important topics."


# ============================================================================
# VectorStore Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_vector_store() -> AsyncMock:
    """
    Create a mock VectorStore for testing.

    Provides async mock methods for:
    - similarity_search
    - get_related_content
    - generate_embedding

    Returns:
        Mock VectorStore instance
    """
    mock_store = AsyncMock()

    # Mock similarity_search
    mock_store.similarity_search = AsyncMock(
        return_value=[
            {
                "id": 1,
                "content": "This is a relevant chunk from the document.",
                "page_number": 3,
                "score": 0.95,
                "metadata": {"related_images": [1], "related_tables": [1]}
            },
            {
                "id": 2,
                "content": "Another relevant chunk discussing similar topics.",
                "page_number": 5,
                "score": 0.87,
                "metadata": {"related_images": [], "related_tables": []}
            }
        ]
    )

    # Mock get_related_content
    mock_store.get_related_content = AsyncMock(
        return_value={
            "images": [
                {
                    "id": 1,
                    "file_path": "/uploads/images/figure1.png",
                    "page_number": 3,
                    "caption": "Figure 1: Overview"
                }
            ],
            "tables": [
                {
                    "id": 1,
                    "image_path": "/uploads/tables/table1.png",
                    "page_number": 5,
                    "caption": "Table 1: Summary",
                    "data": {"headers": ["Col1", "Col2"], "rows": [["A", "B"]]}
                }
            ]
        }
    )

    # Mock generate_embedding
    mock_store.generate_embedding = AsyncMock(
        return_value=[0.1] * 348  # Simulate 348-dimensional embedding
    )

    return mock_store


# ============================================================================
# Test Data Factory Fixtures
# ============================================================================

@pytest.fixture
def sample_conversation(db_session: Session, sample_document):
    """
    Create a sample conversation for testing.
    """
    from app.models.conversation import Conversation

    conversation = Conversation(
        title="Test Conversation",
    )
    conversation.documents.append(sample_document)
    db_session.add(conversation)
    db_session.commit()
    db_session.refresh(conversation)

    return conversation


@pytest.fixture
def sample_messages(db_session: Session, sample_conversation):
    """
    Create sample messages for a conversation.
    """
    from app.models.conversation import Message
    
    messages = [
        Message(
            conversation_id=sample_conversation.id,
            role="user",
            content="What is the main topic of the document?"
        ),
        Message(
            conversation_id=sample_conversation.id,
            role="assistant",
            content="The document primarily discusses advanced topics in machine learning."
        ),
        Message(
            conversation_id=sample_conversation.id,
            role="user",
            content="Can you provide more details?"
        )
    ]

    for msg in messages:
        db_session.add(msg)
    db_session.commit()

    for msg in messages:
        db_session.refresh(msg)

    return messages


@pytest.fixture
def sample_document(db_session: Session):
    """
    Create a sample document for testing.
    """
    from app.models.document import Document

    document = Document(
        filename="test_document.pdf",
        file_path="/uploads/documents/test_document.pdf",
        processing_status="completed",
        total_pages=10,
        text_chunks_count=5,
        images_count=2,
        tables_count=1
    )
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    return document


@pytest.fixture
def sample_chunks(db_session: Session, sample_document: Document) -> list:
    """
    Create sample document chunks for testing.
    """
    chunks = [
        DocumentChunk(
            document_id=sample_document.id,
            content="This is the first chunk of the document about machine learning.",
            page_number=1,
            chunk_index=0,
            embedding=[0.1] * 1536,
            chunk_metadata={"related_images": [1], "related_tables": []}
        ),
        DocumentChunk(
            document_id=sample_document.id,
            content="This is the second chunk discussing neural networks.",
            page_number=2,
            chunk_index=1,
            embedding=[0.2] * 1536,
            chunk_metadata={"related_images": [], "related_tables": [1]}
        ),
        DocumentChunk(
            document_id=sample_document.id,
            content="This is the third chunk covering deep learning.",
            page_number=3,
            chunk_index=2,
            embedding=[0.3] * 1536,
            chunk_metadata={"related_images": [], "related_tables": []}
        )
    ]

    for chunk in chunks:
        db_session.add(chunk)
    db_session.commit()

    for chunk in chunks:
        db_session.refresh(chunk)

    return chunks


@pytest.fixture
def sample_images(db_session: Session, sample_document):
    """
    Create sample document images for testing.
    """
    images = [
        DocumentImage(
            document_id=sample_document.id,
            file_path="/uploads/images/figure1.png",
            page_number=1,
            caption="Figure 1: Architecture Overview",
            width=800,
            height=600,
            image_metadata={"source": "diagram"}
        ),
        DocumentImage(
            document_id=sample_document.id,
            file_path="/uploads/images/figure2.png",
            page_number=3,
            caption="Figure 2: Training Process",
            width=900,
            height=700,
            image_metadata={"source": "plot"}
        )
    ]
    
    for img in images:
        db_session.add(img)
        img.document = sample_document
    db_session.commit()
    
    for img in images:
        db_session.refresh(img)
    
    return images


@pytest.fixture
def sample_tables(db_session: Session, sample_document):
    """
    Create sample document tables for testing.
    """
    tables = [
        DocumentTable(
            document_id=sample_document.id,
            image_path="/uploads/tables/table1.png",
            page_number=2,
            caption="Table 1: Performance Metrics",
            rows=5,
            columns=4,
            data={
                "headers": ["Model", "Accuracy", "Precision", "Recall"],
                "rows": [
                    ["Model A", "0.95", "0.94", "0.96"],
                    ["Model B", "0.92", "0.91", "0.93"],
                    ["Model C", "0.97", "0.96", "0.98"]
                ]
            },
            table_metadata={"source": "experiment"}
        )
    ]
    
    for table in tables:
        db_session.add(table)
        table.document = sample_document
    db_session.commit()
    
    for table in tables:
        db_session.refresh(table)
    
    return tables


# ============================================================================
# Async Event Loop Fixture
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """
    Create an event loop for async tests.
    
    Yields:
        asyncio.AbstractEventLoop
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Auto-patch VectorStore for tests that use mocks
# ============================================================================

@pytest.fixture(autouse=True)
def auto_mock_vector_store_and_settings():
    """
    Auto-mock VectorStore and settings to prevent database connections during test discovery
    and initialization tests.
    
    This fixture patches out VectorStore so tests don't try to connect to real database.
    Tests that need real database can override this.
    """
    from app.services import chat_engine as chat_engine_module
    
    with patch.object(chat_engine_module, 'VectorStore', return_value=MagicMock()), \
         patch.object(chat_engine_module, 'settings') as mock_settings:
        
        # Set minimal settings for LLM initialization
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.OPENAI_MODEL = "gpt-4o-mini"
        mock_settings.USE_GEMINI = False
        
        yield

@pytest.fixture
def create_document_processor(mock_vector_store):
    from app.services.document_processor import DocumentProcessor

    processor = DocumentProcessor()
    processor.vector_store = mock_vector_store

    return processor

@pytest.fixture
def chat_engine_with_mocks(mock_llm_client, mock_vector_store):
    """
    Create a ChatEngine instance with mocked dependencies.
    """
    from app.services.chat_engine import ChatEngine
    
    # Create engine
    engine = ChatEngine()
    
    # Replace dependencies with mocks
    engine.llm_client = mock_llm_client
    engine.vector_store = mock_vector_store
    
    return engine


# ============================================================================
# FastAPI Test Client Fixture
# ============================================================================

@pytest.fixture
def client():
    """
    Create a test client for the FastAPI app.
    """
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)
