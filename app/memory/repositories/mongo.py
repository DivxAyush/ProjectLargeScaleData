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

    async def initialize(self) -> None:
        """Create required indexes. Must be called at startup."""
        try:
            await self._conversations.create_index([("created_at", ASCENDING)])
            await self._messages.create_index(
                [("conversation_id", ASCENDING), ("created_at", ASCENDING)]
            )
        except PyMongoError as exc:
            # We don't swallow the error anymore. Propagate as MemoryStorageError.
            logger.error("Failed to create MongoDB indexes: %s", exc)
            raise MemoryStorageError(f"Failed to create indexes: {exc}") from exc

    async def create_conversation(
        self, conversation_id: str, metadata: dict | None = None
    ) -> Conversation:
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

    async def save_turn(
        self, conversation_id: str, messages: list[StoredMessage]
    ) -> None:
        now = datetime.now(timezone.utc)
        docs = [
            {
                "_id": msg.message_id,
                "conversation_id": msg.conversation_id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at,
                "metadata": msg.metadata or {},
            }
            for msg in messages
        ]
        
        client = self._db.client
        try:
            async with await client.start_session() as session:
                async with session.start_transaction():
                    await self._messages.insert_many(docs, session=session)
                    await self._conversations.update_one(
                        {"_id": conversation_id},
                        {"$set": {"updated_at": now}},
                        session=session,
                    )
        except PyMongoError as exc:
            raise MemoryStorageError(f"Failed to atomically save turn: {exc}") from exc

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
