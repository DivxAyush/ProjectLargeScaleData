"""
app/memory/models.py

Domain models for conversation memory.
These are pure Python dataclasses and have no knowledge of MongoDB or BSON.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class StoredMessage:
    """A single message persisted in the conversation memory."""
    message_id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime
    metadata: dict | None = None


@dataclass
class Conversation:
    """A conversation grouping multiple stored messages."""
    conversation_id: str
    created_at: datetime
    updated_at: datetime
    metadata: dict | None = None


@dataclass(frozen=True)
class ChatResult:
    """Result returned by the ChatService containing the reply and conversation ID."""
    reply: str
    conversation_id: str
