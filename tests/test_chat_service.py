"""
tests/test_chat_service.py

Architecture-verifying tests for ChatService.

These tests prove:
  1. ChatService works with any LLMProvider-shaped object (no SDK required).
  2. LLMProviderError propagates cleanly from the service.
  3. Unexpected provider exceptions are wrapped in LLMError.
  4. The schema→internal-model translation is correct.
"""

import pytest

from app.llm.base import Message
from app.llm.exceptions import LLMError, LLMProviderError
from app.schemas.chat import MessageSchema
from app.services.chat_service import ChatService


# ── Helpers ───────────────────────────────────────────────────────────────────

class _EchoProvider:
    """Returns the last user message content as the reply."""

    async def chat(self, messages: list[Message]) -> str:
        user_messages = [m for m in messages if m.role == "user"]
        return user_messages[-1].content if user_messages else "no user message"


class _InspectingProvider:
    """Captures the messages it receives for assertion."""

    def __init__(self) -> None:
        self.received: list[Message] = []

    async def chat(self, messages: list[Message]) -> str:
        self.received = messages
        return "ok"


class _BrokenProvider:
    """Raises a raw Exception (not LLMProviderError) to test wrapping."""

    async def chat(self, messages: list[Message]) -> str:
        raise RuntimeError("Unexpected SDK crash")


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_service_accepts_any_llm_provider_shaped_object() -> None:
    """
    ChatService must work with any structurally-compatible provider.
    No inheritance from LLMProvider or any SDK import required.
    """
    service = ChatService(provider=_EchoProvider())
    messages = [MessageSchema(role="user", content="ping")]
    reply = await service.chat(messages, user_id="test_user")
    assert reply.reply == "ping"
    assert reply.conversation_id is not None


@pytest.mark.asyncio
async def test_chat_service_translates_schema_to_internal_message() -> None:
    """
    Verify that MessageSchema objects are correctly converted to Message
    dataclasses before being passed to the provider.
    """
    inspector = _InspectingProvider()
    service = ChatService(provider=inspector)

    await service.chat([
        MessageSchema(role="system", content="You are Mili."),
        MessageSchema(role="user", content="Hello"),
    ], user_id="test_user")

    assert len(inspector.received) == 2
    assert all(isinstance(m, Message) for m in inspector.received)
    assert inspector.received[0] == Message(role="system", content="You are Mili.")
    assert inspector.received[1] == Message(role="user", content="Hello")


@pytest.mark.asyncio
async def test_chat_service_propagates_llm_provider_error() -> None:
    """LLMProviderError from the provider must propagate as-is."""
    service = ChatService(provider=_BrokenProvider())

    # Patch _BrokenProvider to raise LLMProviderError instead.
    class _ProviderErrorProvider:
        async def chat(self, messages: list[Message]) -> str:
            raise LLMProviderError("Upstream failure")

    service = ChatService(provider=_ProviderErrorProvider())
    with pytest.raises(LLMProviderError, match="Upstream failure"):
        await service.chat([MessageSchema(role="user", content="Hi")], user_id="test_user")


@pytest.mark.asyncio
async def test_chat_service_wraps_unexpected_exceptions_in_llm_error() -> None:
    """
    A raw non-LLM exception from the provider must be wrapped in LLMError
    so the API layer never has to handle SDK-specific types.
    """
    service = ChatService(provider=_BrokenProvider())
    with pytest.raises(LLMError):
        await service.chat([MessageSchema(role="user", content="Hi")], user_id="test_user")


@pytest.mark.asyncio
async def test_chat_service_is_stateless_across_calls() -> None:
    """Each call to chat() is independent — no shared mutable state."""
    service = ChatService(provider=_EchoProvider())
    r1 = await service.chat([MessageSchema(role="user", content="first")], user_id="test_user")
    r2 = await service.chat([MessageSchema(role="user", content="second")], user_id="test_user")
    assert r1.reply == "first"
    assert r2.reply == "second"
    assert r1.conversation_id != r2.conversation_id

@pytest.mark.asyncio
async def test_chat_service_injects_personal_memory() -> None:
    from unittest.mock import MagicMock, AsyncMock
    from app.memory.personal_models import PersonalMemory
    
    provider = MagicMock()
    provider.chat = AsyncMock(return_value="Reply")
    
    pm_service = AsyncMock()
    pm_service.get_memories.return_value = [
        PersonalMemory(
            memory_id="1", user_id="u", memory_type="preference", key="diet",
            content="I am vegan", source="explicit_user", confidence=1.0,
            created_at=None, updated_at=None
        )
    ]
    
    service = ChatService(provider=provider, personal_memory_service=pm_service)
    
    await service.chat([MessageSchema(role="user", content="Hi")], user_id="explicit_test_user")
    
    pm_service.get_memories.assert_called_once_with(user_id="explicit_test_user", limit=10)
    
    context_passed = provider.chat.call_args[0][0]
    assert len(context_passed) == 2
    assert context_passed[0].role == "system"
    assert "<personal_memory>" in context_passed[0].content
    assert "- I am vegan" in context_passed[0].content
    assert context_passed[1].role == "user"
