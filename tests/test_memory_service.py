"""
tests/test_memory_service.py

Unit tests for MemoryService logic.
"""

import pytest
from app.llm.base import Message
from app.memory.exceptions import ConversationNotFoundError
from tests.conftest import FakeConversationRepository


@pytest.mark.asyncio
async def test_get_or_create_conversation_creates_when_id_is_none(memory_service):
    conv = await memory_service.get_or_create_conversation(None)
    assert conv is not None
    assert conv.conversation_id is not None
    assert len(conv.conversation_id) > 0


@pytest.mark.asyncio
async def test_get_or_create_conversation_returns_existing(memory_service, fake_repository):
    existing = await fake_repository.create_conversation("conv123")
    conv = await memory_service.get_or_create_conversation("conv123")
    assert conv == existing


@pytest.mark.asyncio
async def test_get_or_create_conversation_raises_if_not_found(memory_service):
    with pytest.raises(ConversationNotFoundError):
        await memory_service.get_or_create_conversation("missing123")


@pytest.mark.asyncio
async def test_load_history_returns_empty_list_for_new(memory_service, fake_repository):
    conv = await fake_repository.create_conversation("conv123")
    history = await memory_service.load_history(conv.conversation_id)
    assert history == []


@pytest.mark.asyncio
async def test_save_turn_persists_messages_and_updates_timestamp(memory_service, fake_repository):
    conv = await fake_repository.create_conversation("conv123")
    old_timestamp = conv.updated_at
    
    import asyncio
    await asyncio.sleep(0.001)  # Ensure clock ticks
    
    await memory_service.save_turn(
        conversation_id="conv123",
        user_message=Message(role="user", content="Hi"),
        assistant_reply="Hello"
    )
    
    history = await memory_service.load_history("conv123")
    assert len(history) == 2
    assert history[0] == Message(role="user", content="Hi")
    assert history[1] == Message(role="assistant", content="Hello")
    
    updated_conv = await fake_repository.get_conversation("conv123")
    assert updated_conv.updated_at > old_timestamp
