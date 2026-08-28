"""
app/llm/base.py

Core LLM abstractions. This file contains:

  1. Message — typed internal message representation.
     Used by LLMProvider and ChatService so raw dicts don't scatter
     across the codebase.

  2. LLMProvider — a typing.Protocol that defines the interface every
     provider implementation must satisfy.

IMPORTANT: This file must never import any provider-specific SDK.
Provider SDKs belong exclusively in app/llm/providers/<name>.py.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Message:
    """
    Provider-agnostic message representation.

    role    — One of "user", "assistant", or "system".
    content — The text content of the message.

    Using a frozen dataclass keeps messages immutable and hashable,
    which will be useful when caching or hashing conversation history.
    """

    role: str  # "user" | "assistant" | "system"
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    """
    Structural interface that every LLM provider must satisfy.

    Using typing.Protocol (structural subtyping) means:
      - Provider classes do NOT need to inherit from this class.
      - Any object with a matching `chat` method satisfies the interface.
      - Mock providers in tests require zero imports from the LLM package.

    The chat() method is async because real provider calls are I/O-bound
    network requests. All providers must honour this contract.
    """

    async def chat(self, messages: list[Message]) -> str:
        """
        Send a list of messages to the LLM and return the assistant reply.

        Args:
            messages: Ordered conversation history. At minimum, the last
                      message should have role="user".

        Returns:
            The assistant's reply as a plain string.

        Raises:
            LLMProviderError: On any failure from the underlying provider.
        """
        ...
