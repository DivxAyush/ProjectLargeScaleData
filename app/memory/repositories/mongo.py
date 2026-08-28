"""
app/memory/repositories/mongo.py

MongoDB implementation of the ConversationRepository Protocol.
"""

from datetime import datetime, timezone
import logging

from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import PyMongoError
from pymongo import ASCENDING

from app.memory.exceptions import MemoryStorageError
from app.memory.models import Conversation, StoredMessage
from app.memory.repository import ConversationRepository

logger = logging.getLogger(__name__)


class MongoConversationRepository:
    """
    MongoDB persistence for conversations.
    """

    def __init__(self, db: AsyncDatabase) -> None:
        self._db = db
        self._conversations = db["conversations"]
        self._messages = db["messages"]
        self._indexes_created = False

    async def _ensure_indexes(self) -> None:
        if not self._indexes_created:
            try:
                await self._conversations.create_index([("created_at", ASCENDING)])
                await self._messages.create_index(
                    [("conversation_id", ASCENDING), ("created_at", ASCENDING)]
                )
                self._indexes_created = True
            except PyMongoError as exc:
                logger.error("Failed to create MongoDB indexes: %s", exc)
                # We don't raise here; missing indexes shouldn't block inserts,
                # but it will log the error.
                pass

    async def create_conversation(
        self, conversation_id: str, metadata: dict | None = None
    ) -> Conversation:
        await self._ensure_indexes()
        now = datetime.now(timezone.utc)
        doc = {
            "_id": conversation_id,
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {},
        }
        try:
            await self._conversations.insert_one(doc)
            return Conversation(
                conversation_id=conversation_id,
                created_at=now,
                updated_at=now,
                metadata=metadata,
            )
        except PyMongoError as exc:
            raise MemoryStorageError(f"Failed to create conversation: {exc}") from exc

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        try:
            doc = await self._conversations.find_one({"_id": conversation_id})
            if not doc:
                return None
            return Conversation(
                conversation_id=doc["_id"],
                created_at=doc["created_at"],
                updated_at=doc["updated_at"],
                metadata=doc.get("metadata"),
            )
        except PyMongoError as exc:
            raise MemoryStorageError(f"Failed to retrieve conversation: {exc}") from exc

    async def update_conversation_timestamp(self, conversation_id: str) -> None:
        now = datetime.now(timezone.utc)
        try:
            await self._conversations.update_one(
                {"_id": conversation_id}, {"$set": {"updated_at": now}}
            )
        except PyMongoError as exc:
            raise MemoryStorageError(f"Failed to update timestamp: {exc}") from exc

    async def save_message(self, message: StoredMessage) -> None:
        await self._ensure_indexes()
        doc = {
            "_id": message.message_id,
            "conversation_id": message.conversation_id,
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at,
            "metadata": message.metadata or {},
        }
        try:
            await self._messages.insert_one(doc)
        except PyMongoError as exc:
            raise MemoryStorageError(f"Failed to save message: {exc}") from exc

    async def get_messages(self, conversation_id: str) -> list[StoredMessage]:
        try:
            cursor = self._messages.find({"conversation_id": conversation_id}).sort(
                "created_at", ASCENDING
            )
            messages = []
            async for doc in cursor:
                messages.append(
                    StoredMessage(
                        message_id=doc["_id"],
                        conversation_id=doc["conversation_id"],
                        role=doc["role"],
                        content=doc["content"],
                        created_at=doc["created_at"],
                        metadata=doc.get("metadata"),
                    )
                )
            return messages
        except PyMongoError as exc:
            raise MemoryStorageError(f"Failed to retrieve messages: {exc}") from exc

# Register the class as satisfying the Protocol
# This verifies at runtime that MongoConversationRepository implements the Protocol methods.
# MyPy does this statically.
assert issubclass(MongoConversationRepository, ConversationRepository)
