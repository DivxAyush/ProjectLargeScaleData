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
from app.memory.models import ChatResult
from app.schemas.chat import MessageSchema

logger = logging.getLogger(__name__)


class ChatService:
    """
    Application service that mediates chat interactions.

    The provider is injected at construction time (via FastAPI's DI in
    app/dependencies.py), making the service trivially testable with any
    mock that satisfies the LLMProvider Protocol.
    """

    def __init__(
        self,
        provider: LLMProvider,
        memory_service: "MemoryService | None" = None,
    ) -> None:
        self._provider = provider
        self._memory_service = memory_service

    async def chat(
        self,
        messages: list[MessageSchema],
        conversation_id: str | None = None,
    ) -> ChatResult:
        """
        Process a chat request and return the assistant's reply.

        Args:
            messages: Ordered conversation history from the HTTP request.
            conversation_id: Optional ID of the conversation to load memory for.

        Returns:
            ChatResult containing the reply and conversation ID.

        Raises:
            LLMError: On any LLM-related failure (provider-agnostic).
            MemoryError: On any persistence failure if memory_service is active.
        """
        # Translate HTTP-boundary schemas → internal typed model.
        new_messages = [
            Message(role=m.role, content=m.content) for m in messages
        ]

        if not new_messages or new_messages[-1].role != "user":
            # Just a sanity check, ideally schemas validate this.
            pass

        full_context = []
        resolved_conversation_id = conversation_id

        if self._memory_service:
            # 1. Get or create the conversation
            conversation = await self._memory_service.get_or_create_conversation(conversation_id)
            resolved_conversation_id = conversation.conversation_id
            
            # 2. Load historical context
            history = await self._memory_service.load_history(resolved_conversation_id)
            full_context.extend(history)
        else:
            if resolved_conversation_id is None:
                import uuid
                resolved_conversation_id = str(uuid.uuid4())

        # 3. Append the new messages
        full_context.extend(new_messages)

        logger.info("Sending %d message(s) to provider", len(full_context))

        try:
            reply = await self._provider.chat(full_context)
        except LLMProviderError:
            # Already a clean, provider-agnostic exception — re-raise as-is.
            raise
        except Exception as exc:
            # Unexpected error from the provider — wrap it.
            logger.exception("Unexpected error from LLM provider")
            raise LLMError(f"Unexpected provider error: {exc}") from exc

        logger.info("Provider returned reply (%d chars)", len(reply))

        if self._memory_service:
            # 4. Save the turn (last user message + assistant reply)
            # Find the last user message from the new_messages
            user_msg = new_messages[-1]
            await self._memory_service.save_turn(
                conversation_id=resolved_conversation_id,
                user_message=user_msg,
                assistant_reply=reply,
            )

        return ChatResult(
            reply=reply,
            conversation_id=resolved_conversation_id,
        )
