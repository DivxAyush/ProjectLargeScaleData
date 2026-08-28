"""
app/services/chat_service.py

ChatService — the orchestration layer between the API and the LLM.

Responsibilities:
  - Accept HTTP-boundary schemas (MessageSchema) and translate them to
    the internal Message model before passing them to the provider.
  - Delegate to the injected LLMProvider.
  - Wrap provider-specific exceptions in provider-agnostic LLMError so
    the API layer never has to handle SDK types.

What this service deliberately does NOT do:
  - Import or reference any provider SDK.
  - Handle HTTP concerns (status codes, headers, request/response parsing).
  - Know which concrete provider it is talking to.

Future capabilities that will extend this service:
  - Conversation memory (MongoDB retrieval + persistence).
  - RAG context injection before the provider call.
  - Tool/function-call orchestration loops.
"""

import logging

from app.llm.base import LLMProvider, Message
from app.llm.exceptions import LLMError, LLMProviderError
from app.schemas.chat import MessageSchema

logger = logging.getLogger(__name__)


class ChatService:
    """
    Application service that mediates chat interactions.

    The provider is injected at construction time (via FastAPI's DI in
    app/dependencies.py), making the service trivially testable with any
    mock that satisfies the LLMProvider Protocol.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def chat(self, messages: list[MessageSchema]) -> str:
        """
        Process a chat request and return the assistant's reply.

        Args:
            messages: Ordered conversation history from the HTTP request.

        Returns:
            The assistant's reply as a plain string.

        Raises:
            LLMError: On any LLM-related failure (provider-agnostic).
        """
        # Translate HTTP-boundary schemas → internal typed model.
        internal_messages = [
            Message(role=m.role, content=m.content) for m in messages
        ]

        logger.info("Sending %d message(s) to provider", len(internal_messages))

        try:
            reply = await self._provider.chat(internal_messages)
        except LLMProviderError:
            # Already a clean, provider-agnostic exception — re-raise as-is.
            raise
        except Exception as exc:
            # Unexpected error from the provider — wrap it.
            logger.exception("Unexpected error from LLM provider")
            raise LLMError(f"Unexpected provider error: {exc}") from exc

        logger.info("Provider returned reply (%d chars)", len(reply))
        return reply
