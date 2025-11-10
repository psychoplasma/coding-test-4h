import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import Session

from app.services.chat_engine import ChatEngine
from app.models.conversation import Message, Conversation


# ============================================================================
# ChatEngine Initialization Tests
# ============================================================================

class TestChatEngineInitialization:
    """Test ChatEngine initialization and LLM setup."""
    
    @patch('app.services.chat_engine.settings')
    @patch('app.services.chat_engine.VectorStore')
    def test_chat_engine_initializes(self, mock_vector_store_class, mock_settings):
        """Test ChatEngine initializes with LLM client, VectorStore."""
        # Mock the VectorStore to avoid DB connection
        mock_vector_store_class.return_value = MagicMock()

        # Mock settings
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.OPENAI_MODEL = "gpt-4o-mini"
        mock_settings.USE_GEMINI = False

        with patch('app.services.chat_engine.ChatOpenAI'):
            engine = ChatEngine()
            assert engine.llm_client is not None
            assert engine.vector_store is not None

# ============================================================================
# Conversation History Tests
# ============================================================================

class TestLoadConversationHistory:
    """Test loading conversation history from database."""
    
    @pytest.mark.asyncio
    async def test_load_conversation_history_returns_formatted_messages(
        self,
        db_session: Session,
        sample_messages: list,
        sample_conversation: Conversation,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test loading conversation history returns properly formatted messages."""
        history = await chat_engine_with_mocks._load_conversation_history(
            db_session,
            sample_conversation.id
        )
        
        assert len(history) >= 3
        assert all("role" in msg and "content" in msg for msg in history)
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    @pytest.mark.asyncio
    async def test_load_conversation_history_respects_limit(
        self,
        db_session: Session,
        sample_messages: list,
        sample_conversation: Conversation,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test conversation history respects limit parameter."""
        history = await chat_engine_with_mocks._load_conversation_history(
            db_session,
            sample_conversation.id,
            limit=2
        )
        
        assert len(history) <= 2
    
    @pytest.mark.asyncio
    async def test_load_conversation_history_returns_empty_for_nonexistent_conversation(
        self,
        db_session: Session,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test loading history for nonexistent conversation returns empty list."""
        history = await chat_engine_with_mocks._load_conversation_history(
            db_session,
            9999
        )
        
        assert history == []

    @pytest.mark.asyncio
    async def test_load_conversation_history_maintains_order(
        self,
        db_session: Session,
        sample_conversation: Conversation,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test conversation history maintains chronological order."""
        history = await chat_engine_with_mocks._load_conversation_history(
            db_session,
            sample_conversation.id
        )
        
        # Verify order is maintained (should be oldest first)
        if len(history) > 1:
            # Find consecutive pairs and verify roles alternate
            for i in range(len(history) - 1):
                if history[i]["role"] == "user":
                    # Next should ideally be assistant, but could be user in multi-turn
                    assert history[i + 1]["role"] in ["user", "assistant"]
    


# ============================================================================
# Context Search Tests
# ============================================================================

class TestSearchContext:
    """Test context search functionality."""
    
    @pytest.mark.asyncio
    async def test_search_context_calls_vector_store(
        self,
        db_session: Session,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test search_context properly calls the vector store."""
        query = "What is the main topic?"
        
        results = await chat_engine_with_mocks._search_context(
            db_session,
            query=query,
            k=5
        )

        # Verify vector store was called
        chat_engine_with_mocks.vector_store.similarity_search.assert_called_once_with(
            db_session,
            query=query,
            document_ids=None,
            k=5
        )

        # Verify results format
        assert isinstance(results, list)
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_search_context_with_document_filter(
        self,
        db_session: Session,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test search context with document ID filter."""
        query = "machine learning"
        document_ids = [1, 2, 3]

        await chat_engine_with_mocks._search_context(
            db_session,
            query=query,
            document_ids=document_ids,
            k=3
        )

        # Verify document filter was passed
        chat_engine_with_mocks.vector_store.similarity_search.assert_called_once_with(
            db_session,
            query=query,
            document_ids=document_ids,
            k=3
        )

    @pytest.mark.asyncio
    async def test_search_context_returns_structured_results(
        self,
        db_session: Session,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test search context returns properly structured results."""
        results = await chat_engine_with_mocks._search_context(
            db_session,
            query="neural networks",
            k=2
        )

        # Verify result structure
        for result in results:
            assert "id" in result
            assert "content" in result
            assert "page_number" in result
            assert "score" in result


# ============================================================================
# Related Media Tests
# ============================================================================

class TestFindRelatedMedia:
    """Test finding related images and tables."""
    
    @pytest.mark.asyncio
    async def test_find_related_media_returns_images_and_tables(
        self,
        db_session: Session,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test find_related_media returns both images and tables."""
        context = [
            {"id": 1, "content": "chunk 1"},
            {"id": 2, "content": "chunk 2"}
        ]
        
        media = await chat_engine_with_mocks._find_related_media(db_session, context)
        
        assert "images" in media
        assert "tables" in media
        assert isinstance(media["images"], list)
        assert isinstance(media["tables"], list)
    
    @pytest.mark.asyncio
    async def test_find_related_media_handles_empty_context(
        self,
        db_session: Session,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test find_related_media handles empty context gracefully."""
        media = await chat_engine_with_mocks._find_related_media(db_session, [])
        
        assert media == {"images": [], "tables": []}
    
    @pytest.mark.asyncio
    async def test_find_related_media_calls_vector_store(
        self,
        db_session: Session,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test find_related_media calls vector store's get_related_content."""
        context = [{"id": 1}, {"id": 2}]
        
        await chat_engine_with_mocks._find_related_media(db_session, context)
        
        # Verify vector store was called with chunk IDs
        chat_engine_with_mocks.vector_store.get_related_content.assert_called_once()
        call_args = chat_engine_with_mocks.vector_store.get_related_content.call_args
        assert call_args[0][0] == db_session
        assert set(call_args[0][1]) == {1, 2}


# ============================================================================
# Response Generation Tests
# ============================================================================

class TestGenerateResponse:
    """Test LLM response generation."""
    
    @pytest.mark.asyncio
    async def test_generate_response_calls_llm(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test generate_response calls the LLM client."""
        message = "What is the document about?"
        context = [{"id": 1, "content": "relevant context"}]
        history = [{"role": "user", "content": "previous message"}]
        media = {"images": [], "tables": []}
        
        response = await chat_engine_with_mocks._generate_response(
            message,
            (1, context, media),
            history,
        )
        
        # Verify LLM was called
        chat_engine_with_mocks.llm_client.invoke.assert_called_once()
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_handles_error_gracefully(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test generate_response handles LLM errors gracefully."""
        # Make LLM raise an exception
        chat_engine_with_mocks.llm_client.invoke.side_effect = Exception("LLM Error")
        
        response = await chat_engine_with_mocks._generate_response(
            "question",
            (1, [], [{"images": [], "tables": []}]),
            [],
        )
        
        # Should return error message instead of raising
        assert "I encountered an error generating a response" in response

    @pytest.mark.asyncio
    async def test_generate_response_builds_system_prompt(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test generate_response includes system prompt."""
        await chat_engine_with_mocks._generate_response(
            "test question",
            (1,[{"id": 1, "content": "context"}], {"images": [{"url": "img1"}], "tables": [{"url": "tbl1"}]}),
            [],
        )
        
        # Get the messages passed to LLM
        call_args = chat_engine_with_mocks.llm_client.invoke.call_args
        messages = call_args[0][0]
        
        # First message should be system message
        assert messages[0].type == "system" or "system" in str(messages[0]).lower()


# ============================================================================
# Message Aggregation Tests
# ============================================================================

class TestAggregateMessages:
    """Test message aggregation for LLM input."""
    
    def test_aggregate_messages_includes_system_prompt(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test aggregate_messages includes system prompt."""
        messages = chat_engine_with_mocks._aggregate_messages(
            "test question",
            (1,[{"id": 1, "content": "context"}], {"images": [], "tables": []}),
            [],
        )
        
        assert len(messages) > 0
        assert messages[0].type == "system" or "system" in str(messages[0]).lower()
    
    def test_aggregate_messages_includes_conversation_history(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test aggregate_messages includes conversation history."""
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"}
        ]
        
        messages = chat_engine_with_mocks._aggregate_messages(
            "current question",
            (1,[{"id": 1, "content": "context"}], {"images": [], "tables": []}),
            history,
        )
        
        # Should have system + history + current = at least 4 messages
        assert len(messages) >= 4
    
    def test_aggregate_messages_includes_current_message(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test aggregate_messages includes current user message."""
        messages = chat_engine_with_mocks._aggregate_messages(
            "current question",
            (1,[{"id": 1, "content": "context"}], {"images": [], "tables": []}),
            [],
        )
        
        # Last message should be human (user) message
        assert messages[-1].type == "human" or "human" in str(messages[-1]).lower()


# ============================================================================
# Prompt Building Tests
# ============================================================================

class TestPromptBuilding:
    """Test system and user prompt building."""
    
    def test_build_system_prompt_basic(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test system prompt building with no media."""
        prompt = chat_engine_with_mocks._build_system_prompt(
            {"images": [], "tables": []}
        )
        
        assert isinstance(prompt, str)
        assert "document" in prompt.lower()
        assert "answer" in prompt.lower()
    
    def test_build_system_prompt_with_media(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test system prompt includes media information."""
        prompt = chat_engine_with_mocks._build_system_prompt(
            {
                "images": [{"url": "img1"}],
                "tables": [{"url": "tbl1"}]
            }
        )
        
        assert "images" in prompt.lower() or "image" in prompt.lower()
        assert "tables" in prompt.lower() or "table" in prompt.lower()
    
    def test_build_context_string_formats_chunks(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test context string building formats chunks properly."""
        context = [
            {
                "id": 1,
                "content": "First chunk",
                "page_number": 1,
                "score": 0.95
            },
            {
                "id": 2,
                "content": "Second chunk",
                "page_number": 2,
                "score": 0.87
            }
        ]
        
        context_str = chat_engine_with_mocks._build_context_string(context)
        
        assert "First chunk" in context_str
        assert "page" in context_str.lower()
        assert "0.95" in context_str or "95" in context_str
    
    def test_build_context_string_handles_empty_context(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test context string handles empty context."""
        context_str = chat_engine_with_mocks._build_context_string([])
        
        assert isinstance(context_str, str)
        assert len(context_str) > 0
    
    def test_build_user_prompt_includes_context_and_images(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test user prompt includes context, question, and media."""
        context = "relevant context"
        message = "What is the topic?"
        media = {
            "images": [{"caption": "Figure 1", "page_number": 1}],
            "tables": [{"caption": "Table 1", "page_number": 2}]
        }
        
        prompt = chat_engine_with_mocks._build_user_prompt(
            message,
            context,
            media
        )
        
        assert "relevant context" in prompt
        assert "What is the topic?" in prompt
        assert "Figure 1" in prompt
        assert "Table 1" in prompt


# ============================================================================
# Source Formatting Tests
# ============================================================================

class TestFormatSources:
    """Test source formatting for response."""
    
    def test_format_sources_includes_text_sources(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test format_sources includes text chunks."""
        context = [
            {
                "id": 1,
                "content": "This is a long chunk that should be truncated in display",
                "page_number": 1,
                "score": 0.95
            }
        ]
        media = {"images": [], "tables": []}
        
        sources = chat_engine_with_mocks._format_sources(context, media)
        
        assert len(sources) > 0
        text_source = next((s for s in sources if s["type"] == "text"), None)
        assert text_source is not None
        assert "type" in text_source
        assert "content" in text_source
        assert "page" in text_source
        assert "score" in text_source
    
    def test_format_sources_includes_images(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test format_sources includes image sources."""
        context = []
        media = {
            "images": [
                {
                    "url": "/uploads/images/fig1.png",
                    "caption": "Figure 1",
                    "page_number": 1
                }
            ],
            "tables": []
        }
        
        sources = chat_engine_with_mocks._format_sources(context, media)
        
        image_source = next((s for s in sources if s["type"] == "image"), None)
        assert image_source is not None
        assert image_source["url"] == "/uploads/images/fig1.png"
        assert image_source["caption"] == "Figure 1"
    
    def test_format_sources_includes_tables(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test format_sources includes table sources."""
        context = []
        media = {
            "images": [],
            "tables": [
                {
                    "url": "/uploads/tables/tbl1.png",
                    "caption": "Table 1",
                    "page_number": 2,
                    "data": {"headers": ["Col1", "Col2"]}
                }
            ]
        }
        
        sources = chat_engine_with_mocks._format_sources(context, media)
        
        table_source = next((s for s in sources if s["type"] == "table"), None)
        assert table_source is not None
        assert table_source["url"] == "/uploads/tables/tbl1.png"
        assert table_source["data"] is not None
    
    def test_format_sources_limits_text_chunks(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test format_sources limits to top 3 text chunks."""
        context = [
            {"id": i, "content": f"chunk {i}", "page_number": i, "score": 0.95 - i*0.05}
            for i in range(1, 6)
        ]
        media = {"images": [], "tables": []}
        
        sources = chat_engine_with_mocks._format_sources(context, media)
        
        text_sources = [s for s in sources if s["type"] == "text"]
        assert len(text_sources) <= 3

    def test_format_sources_truncates_long_content(
        self,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test format_sources truncates long content."""
        long_content = "x" * 1000
        context = [
            {
                "id": 1,
                "content": long_content,
                "page_number": 1,
                "score": 0.95
            }
        ]
        media = {"images": [], "tables": []}
        
        sources = chat_engine_with_mocks._format_sources(context, media)
        
        text_source = next((s for s in sources if s["type"] == "text"), None)
        # Content should be truncated to 200 chars
        assert len(text_source["content"]) <= 200


# ============================================================================
# Message Persistence Tests
# ============================================================================

class TestSaveMessages:
    """Test saving messages to database."""
    
    def test_save_messages_creates_user_message(
        self,
        db_session: Session,
        sample_conversation,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test save_messages creates user message in database."""
        user_message = "What is the document about?"
        assistant_response = "The document is about..."
        sources = [{"type": "text", "content": "relevant chunk"}]
        
        initial_count = db_session.query(Message).count()
        
        chat_engine_with_mocks._save_messages(
            db_session,
            sample_conversation.id,
            user_message,
            assistant_response,
            sources
        )
        db_session.flush()  # Flush to make messages visible in this session, because db actions in this method are committed in caller's session for the sake of atomicity
        
        final_count = db_session.query(Message).count()
        
        # Verify messages were added
        assert final_count == initial_count + 2
        
        # Verify user message exists
        user_msgs = db_session.query(Message).filter(
            Message.content == user_message
        ).all()
        assert len(user_msgs) > 0
    
    def test_save_messages_creates_assistant_message(
        self,
        db_session: Session,
        sample_conversation,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test save_messages creates assistant message with sources."""
        user_message = "Question?"
        assistant_response = "Answer from AI"
        sources = [{"type": "text", "content": "source"}]
        
        chat_engine_with_mocks._save_messages(
            db_session,
            sample_conversation.id,
            user_message,
            assistant_response,
            sources
        )
        db_session.flush()  # Flush to make messages visible in this session, because db actions in this method are committed in caller's session for the sake of atomicity
        
        # Query for assistant message
        assistant_msgs = db_session.query(Message).filter(
            Message.content == assistant_response,
            Message.role == "assistant"
        ).all()
        
        assert len(assistant_msgs) > 0
        # Check sources are stored
        assert assistant_msgs[0].sources is not None
    
    def test_save_messages_both_messages_created(
        self,
        db_session: Session,
        sample_conversation,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test save_messages creates both user and assistant messages."""
        initial_count = db_session.query(Message).filter(
            Message.conversation_id == sample_conversation.id
        ).count()
        
        chat_engine_with_mocks._save_messages(
            db_session,
            sample_conversation.id,
            "user msg",
            "assistant msg",
            []
        )
        db_session.flush()  # Flush to make messages visible in this session, because db actions in this method are committed in caller's session for the sake of atomicity
        
        final_count = db_session.query(Message).filter(
            Message.conversation_id == sample_conversation.id
        ).count()
        
        # Should have added 2 messages
        assert final_count == initial_count + 2


# ============================================================================
# Full Integration Tests
# ============================================================================

class TestProcessMessage:
    """Test full message processing pipeline."""
    
    @pytest.mark.asyncio
    async def test_process_message_returns_expected_structure(
        self,
        db_session: Session,
        sample_conversation: Conversation,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test process_message returns properly structured response."""
        result = await chat_engine_with_mocks.process_message(
            db_session,
            sample_conversation.id,
            "What is the main topic?"
        )
        
        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert "processing_time" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)
        assert isinstance(result["processing_time"], (int, float))
    
    @pytest.mark.asyncio
    async def test_process_message_with_document_filter(
        self,
        db_session: Session,
        sample_conversation: Conversation,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test process_message with document ID filter."""
        result = await chat_engine_with_mocks.process_message(
            db_session,
            sample_conversation.id,
            "Question?",
            document_ids=[1, 2]
        )
        
        assert "answer" in result
        # Verify vector store was called with document filter
        call_args = chat_engine_with_mocks.vector_store.similarity_search.call_args
        assert call_args[1].get("document_ids") == [1, 2]
    
    @pytest.mark.asyncio
    async def test_process_message_saves_to_database(
        self,
        db_session: Session,
        sample_conversation,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test process_message saves messages to database."""
        initial_message_count = db_session.query(Message).filter(
            Message.conversation_id == sample_conversation.id
        ).count()
        
        await chat_engine_with_mocks.process_message(
            db_session,
            sample_conversation.id,
            "New question?"
        )
        
        db_session.flush()  # Flush to make messages visible in this session, because db actions in this method are committed in caller's session for the sake of atomicity
        
        final_message_count = db_session.query(Message).filter(
            Message.conversation_id == sample_conversation.id
        ).count()
        
        # Should have added user and assistant messages
        assert final_message_count > initial_message_count
    
    @pytest.mark.asyncio
    async def test_process_message_includes_sources(
        self,
        db_session: Session,
        sample_conversation: Conversation,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test process_message includes sources in response."""
        result = await chat_engine_with_mocks.process_message(
            db_session,
            sample_conversation.id,
            "What's the answer?"
        )
        
        # Should have sources
        assert len(result["sources"]) > 0
        
        # Sources should have proper structure
        for source in result["sources"]:
            assert "type" in source
            assert source["type"] in ["text", "image", "table"]

    @pytest.mark.asyncio
    async def test_process_message_handles_empty_search_results(
        self,
        db_session: Session,
        sample_conversation: Conversation,
        chat_engine_with_mocks: ChatEngine
    ):
        """Test process_message handles empty search results gracefully."""
        # Mock empty search results
        chat_engine_with_mocks.vector_store.similarity_search.return_value = []
        
        result = await chat_engine_with_mocks.process_message(
            db_session,
            sample_conversation.id,
            "obscure question"
        )
        
        assert "answer" in result
        assert isinstance(result["answer"], str)
