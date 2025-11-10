# Backend - Multimodal Document Chat System

A FastAPI-based backend service for processing PDF documents and providing intelligent multimodal chat capabilities. The system extracts text, images, and tables from documents, stores them in a vector database, and enables users to query documents with contextual responses that include relevant images and tables.

## Project Overview

This backend service implements a complete document processing and retrieval-augmented generation (RAG) system with multimodal capabilities. Users can:

- **Upload PDF documents** for automatic processing
- **Extract content** including text, images, and tables using Docling
- **Perform semantic search** across documents using embeddings
- **Chat multimodally** - ask questions and receive answers with relevant images and tables
- **Maintain conversations** with full message history and context

### Key Architecture Highlights

- **FastAPI**: Modern, fast web framework for building APIs
- **PostgreSQL + pgvector**: Vector database for semantic search
- **LangChain**: RAG orchestration and LLM integration
- **Docling**: Advanced PDF document parsing and content extraction
- **Redis**: Caching and session management
- **Uvicorn**: ASGI server for high-performance async execution

## Tech Stack

### Core Framework
- **FastAPI** - High-performance web framework
- **Uvicorn** - ASGI server with async support
- **Pydantic** - Data validation and settings management

### Database & Vector Store
- **PostgreSQL** `15.x` - Relational database with pgvector extension
- **pgvector**  - Vector database for embeddings
- **SQLAlchemy** `v2` - ORM for database operations

### Document Processing
- **Docling** - Advanced PDF parsing and content extraction
- **Docling-core** - Core extraction utilities with chunking
- **EasyOCR** - Optical character recognition
- **RapidOCR** - Fast OCR processing
- **Pillow** - Image processing

### AI/ML & RAG
- **LangChain** - LLM orchestration framework
- **LangChain-OpenAI** - OpenAI integration
- **LangChain-Google-VertexAI** - Google Gemini integration
- **Sentence-Transformers** - Local embeddings

## Setup Instructions

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (for dependency management)
- PostgreSQL with pgvector extension

### Option 1: Docker Setup (Recommended)

#### 1. Clone and Navigate to Backend
```bash
cd backend
```

#### 2. Configure Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```bash
# Edit .env file with your credentials
nano .env  # or use your preferred editor
```

#### 3. Start Docker Compose
From the project root directory:
```bash
docker-compose up -d
```

This will start:
- **PostgreSQL** on `localhost:9432`
- **Backend API** on `localhost:8000`
- **Frontend** on `localhost:3000`

#### 4. Verify Services
Check if all services are healthy:
```bash
docker-compose ps

# Check backend logs
docker-compose logs -f backend

# Check frontend logs
docker-compose logs -f frontend
```

#### 5. Access the API
- **Swagger UI**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health
- **Frontend**: http://localhost:3000

### Option 2: Local Development Setup

#### 1. Install Python Dependencies
```bash
cd backend

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies with uv
uv sync
```

#### 2. Setup Database
Ensure PostgreSQL with pgvector is running:
```bash
# Using Docker
docker run -d \
  --name postgres-pgvector \
  -e POSTGRES_USER=docuser \
  -e POSTGRES_PASSWORD=docpass \
  -e POSTGRES_DB=docdb \
  -p 9432:5432 \
  pgvector/pgvector:pg15
```

#### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

#### 5. Start Backend Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

---

## Environment Variables

Create a `.env` file in the `backend/` directory with the following configuration:

```env
# ==================== Database ====================
# PostgreSQL with pgvector extension
DATABASE_URL=postgresql+psycopg2://docuser:docpass@localhost:5432/docdb

# ==================== LLM Configuration ====================
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Google Gemini Configuration (Alternative)
USE_GEMINI=False
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_EMBEDDING_MODEL=text-multilingual-embedding-002
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_PROJECT_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json

# ==================== LLM Settings ====================
LLM_MAX_OUTPUT_TOKENS=1024

# ==================== Upload Settings ====================
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=52428800  # 50 MB in bytes

# ==================== Vector Store Settings ====================
EMBEDDING_DIMENSION=384  # Max 384, DocumentChunk.embeedding column is created with 384, if you want to change this you should change this value in corresponding db model schema. And re-create the database.
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5

# ==================== Logging ====================
LOG_FORMAT=[%(levelname)s] %(asctime)s [%(name)s] %(filename)s:%(lineno)d: %(message)s
LOG_LEVEL=INFO
```

### Environment Variable Guide

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | `sk-...` |
| `OPENAI_MODEL` | GPT model to use | `gpt-4o-mini`, `gpt-4o` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+psycopg2://user:pass@host:port/db` |
| `USE_GEMINI` | Use Google Gemini instead of OpenAI | `True`, `False` |
| `EMBEDDING_DIMENSION` | Vector embedding size | `384` (max) |
| `CHUNK_SIZE` | Text chunk size for processing | `1000` |
| `TOP_K_RESULTS` | Number of search results | `5` |


## API

### Document Processing
- **PDF Upload & Validation** - Support for PDF files up to 50MB
- **Advanced Extraction** - Text, images, and tables extracted using Docling
- **OCR Support** - EasyOCR and RapidOCR for scanned documents
- **Chunking & Vectorization** - Intelligent text chunking with embeddings
- **Background Processing** - Asynchronous document processing via background tasks

### Vector Database
- **Semantic Search** - Vector similarity search using pgvector
- **Embedding Generation** - Support for OpenAI and HuggingFace embeddings
- **Efficient Storage** - Optimized vector storage and retrieval

### Chat & RAG
- **Multimodal Responses** - Answers with related text, images, and tables
- **Multi-turn Conversations** - Maintain conversation context across turns
- **Source Attribution** - Track and return sources for all answers
- **Dual LLM Support** - OpenAI GPT-4 and Google Gemini integration
- **Context Windows** - Configurable token limits and output sizes

### API Features
- **RESTful API** - Full REST API with standard HTTP methods
- **Async/Await** - Non-blocking async operations throughout
- **CORS Support** - Cross-origin requests enabled for frontend
- **Request Validation** - Pydantic models for all requests/responses
- **Pagination** - Support for paginated list endpoints
- **Error Handling** - Comprehensive error messages and status codes

### Database
- **SQLAlchemy ORM** - Robust object-relational mapping
- **Alembic Migrations** - Version-controlled schema management
- **Transaction Support** - ACID compliance with PostgreSQL
- **Relationships** - Proper foreign keys and relationships

### Developer Experience
- **Swagger UI** - Interactive API documentation at `/docs`
- **Structured Logging** - Colored console output with context
- **Hot Reload** - Development server with auto-reload
- **Docker Support** - Multi-stage Docker build for optimization


## Known Limitations

1. **Embedding Dimension** - Fixed at 384 dimensions for HuggingFace models; changing requires database recreation

2. **LLM Context Window** - Limited by the chosen LLM model

3. **Concurrent Processing** - Large documents may take considerable time to process; consider scaling workers

4. **Vector Search** - Top-K results are fixed; no dynamic result count adjustment

5. **Language Support** - Primarily optimized for English; multi-language support depends on LLM capabilities

## Future Improvements

- [ ] **Batch Processing** - Process multiple documents in parallel
- [ ] **Webhooks** - Notify external systems on document processing completion
- [ ] **Query Optimization** - Optimize vector search with indexing
- [ ] **Multi-language Support** - Support for documents in multiple languages
- [ ] **LLM resopnse streaming** - Instead of showing loading graphics, stream llm output tokens as they are generated
- [ ] **Hybrid Search** - Combine vector(semantic) and keyword search(lexical) with a fusion strategy like Reciprocal Rank Fusion algorithm
- [ ] **Relevance Check** - Elemination vector search results with LLM relevance evaluation for the given query and the search output

## Testing

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_documents.py -v
```

### Run with Coverage
```bash
pytest --cov=app --cov-report=html
```
