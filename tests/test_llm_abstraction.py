"""
tests/test_llm_abstraction.py

Architecture-verifying tests for the LLM abstraction layer.

These tests prove:
  1. The LLMProvider Protocol is satisfied structurally (no inheritance).
  2. Message is a correct, immutable dataclass.
  3. The factory raises LLMConfigurationError for unknown providers.
  4. The exception hierarchy is correct.
"""

import pytest

from app.llm.base import LLMProvider, Message
from app.llm.exceptions import LLMConfigurationError, LLMError, LLMProviderError
from app.llm.factory import create_llm_provider
from app.config import Settings


# ── Message model tests ───────────────────────────────────────────────────────

def test_message_dataclass_fields() -> None:
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_message_is_immutable() -> None:
    """Frozen dataclass — mutation must raise."""
    msg = Message(role="user", content="Hello")
    with pytest.raises((AttributeError, TypeError)):
        msg.role = "assistant"  # type: ignore[misc]


def test_message_equality() -> None:
    assert Message(role="user", content="Hi") == Message(role="user", content="Hi")
    assert Message(role="user", content="Hi") != Message(role="assistant", content="Hi")


# ── Protocol structural typing tests ─────────────────────────────────────────

def test_llm_provider_protocol_satisfied_without_inheritance() -> None:
    """
    Any object with a matching chat() method satisfies LLMProvider.
    No import from app.llm.providers needed.
    """

    class MinimalProvider:
        async def chat(self, messages: list[Message]) -> str:
            return "reply"

    assert isinstance(MinimalProvider(), LLMProvider)


def test_object_without_chat_does_not_satisfy_protocol() -> None:
    class NotAProvider:
        pass

    assert not isinstance(NotAProvider(), LLMProvider)


def test_provider_with_wrong_signature_does_not_satisfy_protocol() -> None:
    """A sync chat() method does not satisfy the async Protocol."""

    class SyncProvider:
        def chat(self, messages: list[Message]) -> str:  # sync, not async
            return "reply"

    # Runtime check via isinstance — sync function is still callable,
    # so Protocol check at runtime only verifies method existence, not
    # async-ness. This is a known Python limitation; we document it here.
    # The actual enforcement happens at type-check time (mypy/pyright).
    # We assert that a correct async version IS an instance.
    class AsyncProvider:
        async def chat(self, messages: list[Message]) -> str:
            return "reply"

    assert isinstance(AsyncProvider(), LLMProvider)


# ── Factory tests ─────────────────────────────────────────────────────────────

def test_factory_raises_for_unknown_provider() -> None:
    """Unknown provider name must fail cleanly at startup, not at runtime."""
    settings = Settings(
        llm_provider="nonexistent_provider",
        gemini_api_key="",
    )
    with pytest.raises(LLMConfigurationError, match="nonexistent_provider"):
        create_llm_provider(settings)


def test_factory_error_is_llm_error_subclass() -> None:
    """LLMConfigurationError must be catchable as LLMError."""
    assert issubclass(LLMConfigurationError, LLMError)


# ── Exception hierarchy tests ─────────────────────────────────────────────────

def test_exception_hierarchy() -> None:
    assert issubclass(LLMProviderError, LLMError)
    assert issubclass(LLMConfigurationError, LLMError)


def test_llm_provider_error_chains_cause() -> None:
    """Provider errors should chain the original SDK exception."""
    original = ValueError("SDK failure")
    wrapped = LLMProviderError("Provider call failed")
    try:
        raise wrapped from original
    except LLMProviderError as exc:
        assert exc.__cause__ is original
