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
