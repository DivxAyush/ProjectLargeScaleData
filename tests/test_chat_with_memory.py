"""
tests/test_chat_with_memory.py

Integration tests for ChatService combined with MemoryService.
"""

import pytest

from app.llm.base import Message
from app.llm.exceptions import LLMError
from app.memory.exceptions import MemoryError, MemoryStorageError
from app.schemas.chat import MessageSchema
from app.services.chat_service import ChatService
from tests.conftest import MockLLMProvider, FakeConversationRepository


@pytest.fixture
def chat_service_with_memory(mock_provider, memory_service):
    return ChatService(provider=mock_provider, memory_service=memory_service)


@pytest.mark.asyncio
async def test_chat_with_memory_creates_new_conversation(chat_service_with_memory, fake_repository):
    messages = [MessageSchema(role="user", content="Hello!")]
    
    result = await chat_service_with_memory.chat(messages, conversation_id=None)
    
    assert result.reply == MockLLMProvider.FIXED_REPLY
    assert result.conversation_id is not None
    
    # Verify persistence
    conv = await fake_repository.get_conversation(result.conversation_id)
    assert conv is not None
    
    history = await fake_repository.get_messages(result.conversation_id)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


@pytest.mark.asyncio
async def test_chat_with_memory_uses_existing_conversation(chat_service_with_memory, fake_repository, memory_service):
    # Setup existing conversation
    conv = await fake_repository.create_conversation("conv123")
    await memory_service.save_turn(
        "conv123",
        Message(role="user", content="First turn"),
        "Reply 1"
    )
    
    messages = [MessageSchema(role="user", content="Second turn")]
    result = await chat_service_with_memory.chat(messages, conversation_id="conv123")
    
    assert result.conversation_id == "conv123"
    
    history = await fake_repository.get_messages("conv123")
    assert len(history) == 4
    assert history[-2].content == "Second turn"
    assert history[-1].content == MockLLMProvider.FIXED_REPLY


class ThrowingMemoryService:
    async def get_or_create_conversation(self, conversation_id):
        raise MemoryStorageError("MongoDB disconnected")


@pytest.mark.asyncio
async def test_chat_with_memory_raises_memory_error(mock_provider):
    """MemoryError must propagate out of ChatService."""
    service = ChatService(provider=mock_provider, memory_service=ThrowingMemoryService())
    
    with pytest.raises(MemoryStorageError):
        await service.chat([MessageSchema(role="user", content="Hi")])
