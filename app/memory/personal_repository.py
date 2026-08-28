from typing import Protocol
from .personal_models import PersonalMemory

class PersonalMemoryRepository(Protocol):
    """
    Abstract persistence interface for personal long-term memories.
    """

    async def initialize(self) -> None:
        """Perform one-time initialization (e.g., creating indexes)."""
        ...

    async def create_memory(self, memory: PersonalMemory) -> None:
        """Create a new personal memory. memory_id must be unique."""
        ...

    async def update_memory(self, memory_id: str, content: str, confidence: float | None = None) -> None:
        """Update an existing personal memory. Raises PersonalMemoryNotFoundError if missing."""
        ...

    async def delete_memory(self, memory_id: str) -> None:
        """Delete an existing personal memory. Raises PersonalMemoryNotFoundError if missing."""
        ...

    async def get_memories(
        self, user_id: str, memory_type: str | None = None, limit: int = 10
    ) -> list[PersonalMemory]:
        """
        Retrieve memories for a specific user.
        Must be deterministically ordered by updated_at DESC.
        """
        ...
