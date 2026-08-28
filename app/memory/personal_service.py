import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.memory.personal_models import PersonalMemory, MemoryType, MemorySource
from app.memory.personal_repository import PersonalMemoryRepository

logger = logging.getLogger(__name__)

class PersonalMemoryService:
    def __init__(self, repository: PersonalMemoryRepository) -> None:
        self._repository = repository

    async def create_memory(
        self,
        user_id: str,
        memory_type: MemoryType,
        key: str,
        content: str,
        source: MemorySource,
        confidence: float = 1.0,
    ) -> str:
        """
        Creates a new personal memory. 
        Note: model_suggested memories are supported by the model but 
        should generally not be persisted automatically in V1.2.
        """
        if source == "model_suggested":
            from app.memory.exceptions import MemorySourceRejectedError
            logger.warning("Rejected persistence of model_suggested memory for user %s", user_id)
            raise MemorySourceRejectedError("model_suggested memories cannot be automatically persisted in V1.2")
            
        memory_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        memory = PersonalMemory(
            memory_id=memory_id,
            user_id=user_id,
            memory_type=memory_type,
            key=key,
            content=content,
            source=source,
            confidence=confidence,
            created_at=now,
            updated_at=now,
        )
        
        logger.debug("Saving new personal memory %s for user %s", memory_id, user_id)
        await self._repository.create_memory(memory)
        return memory_id

    async def update_memory(
        self,
        memory_id: str,
        content: str,
        confidence: Optional[float] = None,
    ) -> None:
        """
        Updates an existing personal memory strictly by memory_id.
        """
        logger.debug("Updating personal memory %s", memory_id)
        await self._repository.update_memory(memory_id, content, confidence)

    async def delete_memory(self, memory_id: str) -> None:
        """
        Deletes a personal memory by memory_id.
        """
        logger.debug("Deleting personal memory %s", memory_id)
        await self._repository.delete_memory(memory_id)

    async def get_memories(
        self, user_id: str, memory_type: Optional[MemoryType] = None, limit: int = 10
    ) -> list[PersonalMemory]:
        """
        Retrieves a deterministically ordered, bounded list of memories for a user.
        """
        return await self._repository.get_memories(user_id, memory_type, limit)
