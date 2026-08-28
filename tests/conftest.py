"""
tests/conftest.py

Shared fixtures for the Mili test suite.

Key design decisions:
  - MockLLMProvider satisfies the LLMProvider Protocol WITHOUT importing
    anything from app.llm.providers — proving provider isolation.
  - The test client overrides FastAPI's DI so no real API key is needed.
  - FailingLLMProvider simulates upstream errors for error-path tests.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.llm.base import Message
from app.llm.exceptions import LLMProviderError
from app.main import app
from app.dependencies import get_chat_service
from app.services.chat_service import ChatService
from app.memory.models import Conversation, StoredMessage, ChatResult
from app.memory.personal_models import PersonalMemory
from app.services.chat_service import ChatService
from app.memory.service import MemoryService
from datetime import datetime, timezone


# ── Mock providers ────────────────────────────────────────────────────────────

class MockLLMProvider:
    """
    A minimal object that satisfies the LLMProvider Protocol.

    Notice: does NOT inherit from any class in app.llm.
    This is proof that the Protocol abstraction works correctly.
    """

    FIXED_REPLY = "Hello! I am Mili, your AI assistant."

    async def chat(self, messages: list[Message]) -> str:
        return self.FIXED_REPLY


class FailingLLMProvider:
    """Simulates a provider that always raises an LLMProviderError."""

    async def chat(self, messages: list[Message]) -> str:
        raise LLMProviderError("Simulated upstream failure")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_provider() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def failing_provider() -> FailingLLMProvider:
    return FailingLLMProvider()


@pytest.fixture
def chat_service(mock_provider: MockLLMProvider) -> ChatService:
    """ChatService backed by a mock provider — no SDK required."""
    return ChatService(provider=mock_provider)


@pytest.fixture
def failing_chat_service(failing_provider: FailingLLMProvider) -> ChatService:
    """ChatService backed by a failing provider — for error-path tests."""
    return ChatService(provider=failing_provider)


@pytest.fixture
async def client(chat_service: ChatService) -> AsyncClient:
    """
    Async test client with the real ChatService dependency overridden.
    No API key or network call required.
    """
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
async def failing_client(failing_chat_service: ChatService) -> AsyncClient:
    """Test client that uses a provider that always fails."""
    app.dependency_overrides[get_chat_service] = lambda: failing_chat_service
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


# ── Memory Mock ───────────────────────────────────────────────────────────────

class FakeConversationRepository:
    def __init__(self) -> None:
        self.conversations: dict[str, Conversation] = {}
        self.messages: list[StoredMessage] = []

    async def initialize(self) -> None:
        pass

    async def create_conversation(
        self, conversation_id: str, metadata: dict | None = None
    ) -> Conversation:
        now = datetime.now(timezone.utc)
        conv = Conversation(
            conversation_id=conversation_id,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        self.conversations[conversation_id] = conv
        return conv

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        return self.conversations.get(conversation_id)

    async def update_conversation_timestamp(self, conversation_id: str) -> None:
        if conversation_id in self.conversations:
            self.conversations[conversation_id].updated_at = datetime.now(timezone.utc)

    async def save_turn(self, conversation_id: str, messages: list[StoredMessage]) -> None:
        self.messages.extend(messages)
        await self.update_conversation_timestamp(conversation_id)

    async def get_messages(self, conversation_id: str) -> list[StoredMessage]:
        msgs = [m for m in self.messages if m.conversation_id == conversation_id]
        return sorted(msgs, key=lambda x: x.created_at)


class FakePersonalMemoryRepository:
    def __init__(self) -> None:
        self.memories: dict[str, PersonalMemory] = {}

    async def initialize(self) -> None:
        pass

    async def create_memory(self, memory: PersonalMemory) -> None:
        self.memories[memory.memory_id] = memory

    async def update_memory(self, memory_id: str, content: str, confidence: float | None = None) -> None:
        if memory_id not in self.memories:
            from app.memory.exceptions import PersonalMemoryNotFoundError
            raise PersonalMemoryNotFoundError(f"Not found: {memory_id}")
        self.memories[memory_id].content = content
        if confidence is not None:
            self.memories[memory_id].confidence = confidence
        self.memories[memory_id].updated_at = datetime.now(timezone.utc)

    async def delete_memory(self, memory_id: str) -> None:
        if memory_id not in self.memories:
            from app.memory.exceptions import PersonalMemoryNotFoundError
            raise PersonalMemoryNotFoundError(f"Not found: {memory_id}")
        del self.memories[memory_id]

    async def get_memories(
        self, user_id: str, memory_type: str | None = None, limit: int = 10
    ) -> list[PersonalMemory]:
        results = [
            m for m in self.memories.values()
            if m.user_id == user_id and (memory_type is None or m.memory_type == memory_type)
        ]
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results[:limit]

@pytest.fixture
def fake_personal_repo():
    return FakePersonalMemoryRepository()


@pytest.fixture
def fake_repository() -> FakeConversationRepository:
    return FakeConversationRepository()


@pytest.fixture
def memory_service(fake_repository: FakeConversationRepository) -> MemoryService:
    return MemoryService(repository=fake_repository)
