from datetime import datetime, timezone
import logging
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

from app.db.mongodb import AsyncDatabase
from app.memory.exceptions import MemoryStorageError, PersonalMemoryNotFoundError
from app.memory.personal_models import PersonalMemory
from app.memory.personal_repository import PersonalMemoryRepository

logger = logging.getLogger(__name__)

class MongoPersonalMemoryRepository(PersonalMemoryRepository):
    def __init__(self, db: AsyncDatabase) -> None:
        self._db = db
        self._collection = db["personal_memories"]

    async def initialize(self) -> None:
        """Create required indexes. Must be called at startup."""
        try:
            # Optimize bounded, chronologically ordered retrieval
            await self._collection.create_index([("user_id", ASCENDING), ("updated_at", DESCENDING)])
            # Optimize typed retrieval filtering
            await self._collection.create_index([("user_id", ASCENDING), ("memory_type", ASCENDING), ("updated_at", DESCENDING)])
        except PyMongoError as exc:
            logger.error("Failed to create personal memory MongoDB indexes: %s", exc)
            raise MemoryStorageError(f"Failed to create personal memory indexes: {exc}") from exc

    async def create_memory(self, memory: PersonalMemory) -> None:
        doc = {
            "_id": memory.memory_id,
            "user_id": memory.user_id,
            "memory_type": memory.memory_type,
            "key": memory.key,
            "content": memory.content,
            "source": memory.source,
            "confidence": memory.confidence,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
        }
        try:
            await self._collection.insert_one(doc)
        except PyMongoError as exc:
            raise MemoryStorageError(f"Failed to create personal memory: {exc}") from exc

    async def update_memory(self, memory_id: str, content: str, confidence: float | None = None) -> None:
        updates = {
            "content": content,
            "updated_at": datetime.now(timezone.utc)
        }
        if confidence is not None:
            updates["confidence"] = confidence

        try:
            result = await self._collection.update_one(
                {"_id": memory_id},
                {"$set": updates}
            )
            if result.matched_count == 0:
                raise PersonalMemoryNotFoundError(f"Personal memory {memory_id} not found")
        except PyMongoError as exc:
            raise MemoryStorageError(f"Failed to update personal memory: {exc}") from exc

    async def delete_memory(self, memory_id: str) -> None:
        try:
            result = await self._collection.delete_one({"_id": memory_id})
            if result.deleted_count == 0:
                raise PersonalMemoryNotFoundError(f"Personal memory {memory_id} not found")
        except PyMongoError as exc:
            raise MemoryStorageError(f"Failed to delete personal memory: {exc}") from exc

    async def get_memories(
        self, user_id: str, memory_type: str | None = None, limit: int = 10
    ) -> list[PersonalMemory]:
        query = {"user_id": user_id}
        if memory_type:
            query["memory_type"] = memory_type

        try:
            cursor = self._collection.find(query).sort("updated_at", DESCENDING).limit(limit)
            memories = []
            async for doc in cursor:
                memories.append(
                    PersonalMemory(
                        memory_id=doc["_id"],
                        user_id=doc["user_id"],
                        memory_type=doc["memory_type"],
                        key=doc["key"],
                        content=doc["content"],
                        source=doc["source"],
                        confidence=doc["confidence"],
                        created_at=doc["created_at"],
                        updated_at=doc["updated_at"],
                    )
                )
            return memories
        except PyMongoError as exc:
            raise MemoryStorageError(f"Failed to fetch personal memories: {exc}") from exc
