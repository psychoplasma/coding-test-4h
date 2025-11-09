"""
Chat engine service for multimodal RAG.

Implements:
1. Process user messages
2. Search for relevant context using vector store
3. Find related images and tables
4. Generate responses using LLM
5. Support multi-turn conversations
"""
import logging
import os
import time
from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import asc, select
from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.models.conversation import Message
from app.services.vector_store import VectorStore


logger = logging.getLogger(__name__)


class ChatEngine:
    """
    Multimodal chat engine with RAG (Retrieval-Augmented Generation).
    
    Supports:
    - Multi-turn conversations with history
    - Vector-based similarity search for context retrieval
    - Multimodal responses with images and tables
    - LLM integration (OpenAI or fallback)
    """
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm_client = BaseChatModel | _ConfigurableModel
        self._initialize_llm()
    
    def _initialize_llm(self):
        """
        Initialize LLM client using LangChain.
        
        Supports:
        - OpenAI models via ChatOpenAI
        - Other LLM providers can be added by modifying provider selection logic
        """
        try:
            if settings.OPENAI_API_KEY:
                self.llm_client = ChatOpenAI(
                    name=settings.OPENAI_MODEL,
                    temperature=0.1, # Be factual
                    api_key=settings.OPENAI_API_KEY,
                    max_completion_tokens=500,
                    streaming=False,
                )
                logger.info(f"LLM initialized with OpenAI")
            elif settings.USE_GEMINI:
                self.llm_client = ChatVertexAI(
                    model_name=settings.GEMINI_MODEL,
                    project=settings.GOOGLE_CLOUD_PROJECT,
                    location=settings.GEMINI_MODEL_LOCATION,
                    temperature=0.1, # Be factual
                    max_output_tokens=500,
                    streaming=False,
                )
                logger.info(f"LLM initialized with Gemini")
            else:
                raise ValueError("Either OpenAI API key or Gemini usage flag must be set.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError("LLM initialization failed.") from e
    
    async def process_message(
        self,
        session: Session,
        conversation_id: int,
        message: str,
        document_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate multimodal response.
        
        Implementation steps:
        1. Load conversation history (for multi-turn support)
        2. Search vector store for relevant context
        3. Find related images and tables
        4. Build prompt with context and history
        5. Generate response using LLM
        6. Format response with sources (text, images, tables)
        7. Save user message and assistant response to database
        
        Args:
            conversation_id: Conversation ID
            message: User message
            document_id: Optional document ID to scope search
            
        Returns:
            {
                "answer": "...",
                "sources": [
                    {
                        "type": "text",
                        "content": "...",
                        "page": 3,
                        "score": 0.95
                    },
                    {
                        "type": "image",
                        "url": "/uploads/images/xxx.png",
                        "caption": "Figure 1: ...",
                        "page": 3
                    },
                    {
                        "type": "table",
                        "url": "/uploads/tables/yyy.png",
                        "caption": "Table 1: ...",
                        "page": 5,
                        "data": {...}  # structured table data
                    }
                ],
                "processing_time": 2.5
            }
        """
        start_time = time.time()

        # Step 1: Load conversation history
        history = await self._load_conversation_history(session, conversation_id)

        logger.info(f"Loaded conversation history with {len(history)} messages")

        # Step 2: Search for relevant context
        context = await self._search_context(
            session,
            message,
            document_ids,
            k=settings.TOP_K_RESULTS,
        )

        logger.info(f"Retrieved {len(context)} context chunks from vector store")

        # Step 3: Find related images and tables
        media = await self._find_related_media(session, context)

        logger.info(f"Found {len(media.get('images', []))} images and {len(media.get('tables', []))} tables related to context")

        # Step 4 & 5: Generate response using LLM
        answer = await self._generate_response(message, context, history, media)

        logger.info(f"Generated response from LLM: {answer[:100]}...")

        # Step 6: Format sources for response
        sources = self._format_sources(context, media)

        logger.info(f"Formatted {len(sources)} sources for response")

        # Step 7: Save messages to database
        self._save_messages(session, conversation_id, message, answer, sources)

        processing_time = time.time() - start_time

        return {
            "answer": answer,
            "sources": sources,
            "processing_time": round(processing_time, 2)
        }
    
    async def _load_conversation_history(
        self,
        session: Session,
        conversation_id: int,
        limit: int = 5
    ) -> List[Dict[str, str]]:
        """
        Load recent conversation history.
        
        Loads the last N messages from the conversation and formats them for LLM context.
        Includes both user and assistant messages in chronological order.
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to load (default 5)
            
        Returns:
            List of message dictionaries:
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        """
        # Query recent messages from the conversation chronologically
        # Oldest first
        statement = select(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(
            asc(Message.created_at)
        ).limit(limit)
        messages = session.execute(statement).scalars().all()

        # Format for LLM
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        logger.info(f"Loaded {len(history)} messages from conversation {conversation_id}")
        return history

    async def _search_context(
        self,
        session: Session,
        query: str,
        document_ids: Optional[List[int]] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant context using vector store.
        
        Uses vector similarity search to find the most relevant text chunks
        from the document(s). Filters by document_id if specified.
        
        Args:
            query: Search query text
            document_id: Optional document ID to filter search
            k: Number of results to return (default 5)
            
        Returns:
            List of relevant chunks with metadata:
            [
                {
                    "id": 123,
                    "content": "...",
                    "page_number": 3,
                    "score": 0.95,
                    "metadata": {...}
                },
                ...
            ]
        """
        context = await self.vector_store.similarity_search(
            session,
            query=query,
            document_ids=document_ids,
            k=k
        )

        logger.info(f"Found {len(context)} relevant chunks for query")
        return context
    
    async def _find_related_media(
        self,
        session: Session,
        context: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find related images and tables from context chunks.
        
        Extracts image and table references from chunk metadata and retrieves
        the actual media records from the database. Includes URLs and metadata
        for frontend display.
        
        Args:
            context_chunks: List of context chunks with metadata
            
        Returns:
            Dictionary with images and tables:
            {
                "images": [
                    {
                        "url": "/uploads/images/xxx.png",
                        "caption": "...",
                        "page_number": 3
                    }
                ],
                "tables": [
                    {
                        "url": "/uploads/tables/yyy.png",
                        "caption": "...",
                        "page_number": 5,
                        "data": {...}
                    }
                ]
            }
        """
        try:
            if not context:
                return {"images": [], "tables": []}

            chunk_ids = [chunk["id"] for chunk in context]
            media = await self.vector_store.get_related_content(session, chunk_ids)
            
            # Convert file paths to URLs
            media_with_urls = {
                "images": [
                    {
                        "url": f"/uploads/images/{img['file_path'].split('/')[-1]}",
                        "caption": img.get("caption"),
                        "page_number": img.get("page_number")
                    }
                    for img in media.get("images", [])
                ],
                "tables": [
                    {
                        "url": f"/uploads/tables/{tbl['image_path'].split('/')[-1]}",
                        "caption": tbl.get("caption"),
                        "page_number": tbl.get("page_number"),
                        "data": tbl.get("data")
                    }
                    for tbl in media.get("tables", [])
                ]
            }
            
            return media_with_urls
        
        except Exception as e:
            logger.error(f"Error finding related media: {e}")
            return {"images": [], "tables": []}
    
    async def _generate_response(
        self,
        message: str,
        context: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        media: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Generate response using LLM.
        
        Builds a comprehensive prompt with:
        - System instructions for multimodal responses
        - Conversation history for context
        - Retrieved context from vector search
        - Available images and tables
        
        Then calls the LLM API to generate a response.
        
        Args:
            message: Current user message
            context: Retrieved context chunks
            history: Conversation history
            media: Related images and tables
            
        Returns:
            Generated response text
        """
        try:
            messages = self._aggregate_messages(message, context, history, media)
            response = self.llm_client.invoke(messages)
            return response.content if response else "I'm sorry, I couldn't generate a response."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error generating a response: {str(e)}"

    def _aggregate_messages(
        self,
        message: str,
        context: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        media: Dict[str, List[Dict[str, Any]]]
    ) -> List[BaseMessage]:
        """Aggregate messages for LLM input."""
        # Build system prompt with multimodal instructions
        system_prompt = self._build_system_prompt(media)

        # Build context string from retrieved chunks
        context_str = self._build_context_string(context)

        # Build user prompt with history and current message
        user_prompt = self._build_user_prompt(message, context_str, media)

        # Convert history to LangChain message format
        messages = [SystemMessage(content=system_prompt)]

        for hist_msg in history:
            if hist_msg["role"] == "user":
                messages.append(HumanMessage(content=hist_msg["content"]))
            elif hist_msg["role"] == "assistant":
                messages.append(AIMessage(content=hist_msg["content"]))

        # Add current user message
        messages.append(HumanMessage(content=user_prompt))

        return messages

    def _build_system_prompt(self, media: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Build system prompt for multimodal responses.
        
        Instructs the LLM to:
        - Reference images and tables when available
        - Cite sources
        - Format for good UX
        """
        prompt = """You are a helpful assistant answering questions about documents.

Instructions:
1. Answer questions based on the provided context from the document
2. Be accurate and cite specific information from the context
3. If images or tables are available, reference them in your answer
4. Format your response clearly with sections if needed
5. If you don't know the answer based on the context, say so clearly
6. Keep responses concise but informative"""
        
        if media.get("images"):
            prompt += f"\n7. Available images: {len(media['images'])} image(s) are available to reference"
        
        if media.get("tables"):
            prompt += f"\n8. Available tables: {len(media['tables'])} table(s) are available to reference"
        
        return prompt
    
    def _build_context_string(self, context: List[Dict[str, Any]]) -> str:
        """Build a formatted string from context chunks for the prompt."""
        if not context:
            return "No relevant context found."
        
        context_parts = ["## Document Context:\n"]
        for i, chunk in enumerate(context[:3], 1):  # Use top 3 chunks
            page = chunk.get("page_number", "unknown")
            score = chunk.get("score", 0.0)
            content = chunk.get("content", "")
            context_parts.append(f"[Chunk {i} - Page {page} - Score: {score:.2f}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _build_user_prompt(
        self,
        message: str,
        context: str,
        media: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Build user prompt with context and media information."""
        prompt = f"{context}\n\n## User Question:\n{message}"
        
        if media.get("images"):
            prompt += f"\n\n## Available Images:\n"
            for img in media["images"]:
                caption = img.get("caption", "No caption")
                page = img.get("page_number", "unknown")
                prompt += f"- Image on page {page}: {caption}\n"
        
        if media.get("tables"):
            prompt += f"\n## Available Tables:\n"
            for tbl in media["tables"]:
                caption = tbl.get("caption", "No caption")
                page = tbl.get("page_number", "unknown")
                prompt += f"- Table on page {page}: {caption}\n"
        
        return prompt
    
    def _save_messages(
        self,
        session: Session,
        conversation_id: int,
        user_message: str,
        assistant_response: str,
        sources: List[Dict[str, Any]]
    ) -> None:
        """
        Save user and assistant messages to database.
        
        Args:
            conversation_id: Conversation ID
            user_message: User's message
            assistant_response: Assistant's response
            sources: Sources used in the response
        """
        # Save user message
        user_msg = Message(
            conversation_id=conversation_id,
            role="user",
            content=user_message,
            sources=None
        )
        session.add(user_msg)
        
        # Save assistant message with sources
        assistant_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_response,
            sources=sources
        )
        session.add(assistant_msg)
        logger.info(f"Saved messages to conversation {conversation_id}")

    def _format_sources(
        self,
        context: List[Dict[str, Any]],
        media: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Format sources for response.
        
        Combines text chunks, images, and tables into a unified source list
        for the frontend to display.
        
        Args:
            context: Retrieved context chunks
            media: Related images and tables
            
        Returns:
            List of formatted sources
        """
        sources = []
        
        # Add text sources (top 3)
        for chunk in context[:3]:
            sources.append({
                "type": "text",
                "content": chunk.get("content", "")[:200],  # Truncate for display
                "page": chunk.get("page_number"),
                "score": chunk.get("score", 0.0)
            })
        
        # Add image sources
        for image in media.get("images", []):
            sources.append({
                "type": "image",
                "url": image.get("url"),
                "caption": image.get("caption"),
                "page": image.get("page_number")
            })
        
        # Add table sources
        for table in media.get("tables", []):
            sources.append({
                "type": "table",
                "url": table.get("url"),
                "caption": table.get("caption"),
                "page": table.get("page_number"),
                "data": table.get("data")
            })
        
        return sources
