"""
app/memory/repository.py

Defines the ConversationRepository Protocol.
"""

from typing import Protocol, runtime_checkable

from app.memory.models import Conversation, StoredMessage


@runtime_checkable
class ConversationRepository(Protocol):
    """
    Abstract persistence interface for conversations and their messages.
    """

    async def create_conversation(
        self, conversation_id: str, metadata: dict | None = None
    ) -> Conversation:
        """Create a new conversation."""
        ...

    async def get_conversation(
        self, conversation_id: str
    ) -> Conversation | None:
        """Retrieve a conversation by ID, or None if not found."""
        ...

    async def update_conversation_timestamp(
        self, conversation_id: str
    ) -> None:
        """Update the updated_at timestamp of a conversation."""
        ...

    async def save_message(self, message: StoredMessage) -> None:
        """Persist a single message in the conversation."""
        ...

    async def get_messages(
        self, conversation_id: str
    ) -> list[StoredMessage]:
        """Retrieve all messages for a conversation, ordered chronologically."""
        ...
