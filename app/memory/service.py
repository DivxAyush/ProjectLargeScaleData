"""
app/memory/service.py

Orchestrates memory operations.
Completely DB-agnostic.
"""

import uuid
from datetime import datetime, timezone
import logging

from app.llm.base import Message
from app.memory.exceptions import ConversationNotFoundError
from app.memory.models import Conversation, StoredMessage
from app.memory.repository import ConversationRepository

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Coordinates persistence of conversation turns and retrieval of history.
    """

    def __init__(self, repository: ConversationRepository) -> None:
        self._repository = repository

    async def get_or_create_conversation(
        self, conversation_id: str | None
    ) -> Conversation:
        """
        Return an existing conversation, or create a new one if conversation_id is None.
        If a conversation_id is provided but not found, raises ConversationNotFoundError.
        """
        if not conversation_id:
            new_id = str(uuid.uuid4())
            logger.info("Creating new conversation: %s", new_id)
            return await self._repository.create_conversation(new_id)

        conversation = await self._repository.get_conversation(conversation_id)
        if not conversation:
            logger.warning("Conversation not found: %s", conversation_id)
            raise ConversationNotFoundError(f"Conversation {conversation_id} not found")
        
        return conversation

    async def load_history(self, conversation_id: str) -> list[Message]:
        """
        Load historical messages for a conversation and convert them to
        the LLM provider's internal Message format.
        """
        stored_messages = await self._repository.get_messages(conversation_id)
        logger.debug("Loaded %d historical messages for %s", len(stored_messages), conversation_id)
        return [
            Message(role=sm.role, content=sm.content) for sm in stored_messages
        ]

    async def save_turn(
        self, conversation_id: str, user_message: Message, assistant_reply: str
    ) -> None:
        """
        Save a user message and the corresponding assistant reply, and update
        the conversation's timestamp.
        """
        now = datetime.now(timezone.utc)
        
        user_stored = StoredMessage(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=user_message.role,
            content=user_message.content,
            created_at=now,
        )
        
        assistant_stored = StoredMessage(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_reply,
            created_at=now,
        )

        logger.debug("Saving turn for conversation %s", conversation_id)
        # Atomically save messages and update conversation timestamp
        await self._repository.save_turn(conversation_id, [user_stored, assistant_stored])
