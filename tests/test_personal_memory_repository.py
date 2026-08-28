import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
from pymongo.errors import PyMongoError
from app.memory.repositories.personal_mongo import MongoPersonalMemoryRepository
from app.memory.personal_models import PersonalMemory
from app.memory.exceptions import MemoryStorageError, PersonalMemoryNotFoundError

@pytest.fixture
def mock_db():
    db = MagicMock()
    # Ensure nested collections are AsyncMocks
    db.__getitem__.return_value = AsyncMock()
    return db

@pytest.fixture
def pm_repo(mock_db):
    return MongoPersonalMemoryRepository(mock_db)

@pytest.mark.asyncio
async def test_pm_initialize_success(pm_repo, mock_db):
    await pm_repo.initialize()
    assert mock_db["personal_memories"].create_index.call_count == 2

@pytest.mark.asyncio
async def test_pm_create_memory_success(pm_repo, mock_db):
    pm = PersonalMemory(
        memory_id="123",
        user_id="u1",
        memory_type="preference",
        key="diet",
        content="vegan",
        source="explicit_user",
        confidence=1.0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    await pm_repo.create_memory(pm)
    mock_db["personal_memories"].insert_one.assert_called_once()

@pytest.mark.asyncio
async def test_pm_update_memory_success(pm_repo, mock_db):
    mock_result = MagicMock()
    mock_result.matched_count = 1
    mock_db["personal_memories"].update_one.return_value = mock_result
    
    await pm_repo.update_memory("123", "no dairy", 0.9)
    mock_db["personal_memories"].update_one.assert_called_once()

@pytest.mark.asyncio
async def test_pm_update_memory_not_found(pm_repo, mock_db):
    mock_result = MagicMock()
    mock_result.matched_count = 0
    mock_db["personal_memories"].update_one.return_value = mock_result
    
    with pytest.raises(PersonalMemoryNotFoundError):
        await pm_repo.update_memory("123", "no dairy")

@pytest.mark.asyncio
async def test_pm_delete_memory_success(pm_repo, mock_db):
    mock_result = MagicMock()
    mock_result.deleted_count = 1
    mock_db["personal_memories"].delete_one.return_value = mock_result
    
    await pm_repo.delete_memory("123")
    mock_db["personal_memories"].delete_one.assert_called_once()

@pytest.mark.asyncio
async def test_pm_delete_memory_not_found(pm_repo, mock_db):
    mock_result = MagicMock()
    mock_result.deleted_count = 0
    mock_db["personal_memories"].delete_one.return_value = mock_result
    
    with pytest.raises(PersonalMemoryNotFoundError):
        await pm_repo.delete_memory("123")

class MockCursor:
    def __init__(self, items):
        self.items = items
    def sort(self, *args, **kwargs):
        return self
    def limit(self, *args, **kwargs):
        return self
    async def __aiter__(self):
        for item in self.items:
            yield item

@pytest.mark.asyncio
async def test_pm_get_memories(pm_repo, mock_db):
    docs = [
        {"_id": "1", "user_id": "u1", "memory_type": "preference", "key": "diet", "content": "1", "source": "explicit_user", "confidence": 1.0, "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc)},
        {"_id": "2", "user_id": "u1", "memory_type": "fact", "key": "city", "content": "2", "source": "explicit_user", "confidence": 1.0, "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc)},
    ]
    mock_db["personal_memories"].find = MagicMock(return_value=MockCursor(docs))
    
    results = await pm_repo.get_memories("u1", memory_type="preference", limit=5)
    assert len(results) == 2
    mock_db["personal_memories"].find.assert_called_with({"user_id": "u1", "memory_type": "preference"})
