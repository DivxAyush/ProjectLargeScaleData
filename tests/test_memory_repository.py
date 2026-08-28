"""
tests/test_memory_repository.py

Unit tests for MongoConversationRepository using mock AsyncDatabase.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import PyMongoError
from datetime import datetime, timezone

from app.memory.exceptions import MemoryStorageError
from app.memory.models import StoredMessage
from app.memory.repositories.mongo import MongoConversationRepository
from app.memory.repository import ConversationRepository


@pytest.fixture
def mock_db() -> MagicMock:
    db = MagicMock(spec=AsyncDatabase)
    db.__getitem__.return_value = AsyncMock(spec=AsyncCollection)
    return db


@pytest.fixture
def mongo_repo(mock_db: MagicMock) -> MongoConversationRepository:
    return MongoConversationRepository(db=mock_db)


def test_satisfies_protocol():
    assert issubclass(MongoConversationRepository, ConversationRepository)


@pytest.mark.asyncio
async def test_create_conversation_success(mongo_repo, mock_db):
    mock_db["conversations"].insert_one.return_value = AsyncMock()
    
    conv = await mongo_repo.create_conversation("conv123", {"user_id": "test"})
    
    assert conv.conversation_id == "conv123"
    assert conv.metadata == {"user_id": "test"}
    mock_db["conversations"].insert_one.assert_called_once()
    assert mock_db["conversations"].create_index.assert_called


@pytest.mark.asyncio
async def test_create_conversation_pymongo_error_raises_storage_error(mongo_repo, mock_db):
    mock_db["conversations"].insert_one.side_effect = PyMongoError("DB down")
    
    with pytest.raises(MemoryStorageError, match="Failed to create conversation: DB down"):
        await mongo_repo.create_conversation("conv123")


@pytest.mark.asyncio
async def test_initialize_success(mongo_repo, mock_db):
    await mongo_repo.initialize()
    assert mock_db["conversations"].create_index.assert_called
    assert mock_db["messages"].create_index.assert_called


@pytest.mark.asyncio
async def test_initialize_failure_propagates(mongo_repo, mock_db):
    mock_db["conversations"].create_index.side_effect = PyMongoError("Index error")
    with pytest.raises(MemoryStorageError, match="Failed to create indexes: Index error"):
        await mongo_repo.initialize()


@pytest.mark.asyncio
async def test_save_turn_success(mongo_repo, mock_db):
    msg1 = StoredMessage("msg1", "conv1", "user", "hi", datetime.now(timezone.utc))
    msg2 = StoredMessage("msg2", "conv1", "assistant", "hello", datetime.now(timezone.utc))
    
    mock_client = MagicMock()
    mock_db.client = mock_client
    
    # Mock async context managers for session and transaction
    mock_session = MagicMock()
    mock_client.start_session = AsyncMock(return_value=mock_session)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    
    mock_tx_context = MagicMock()
    mock_tx_context.__aenter__ = AsyncMock(return_value=None)
    mock_tx_context.__aexit__ = AsyncMock(return_value=None)
    mock_session.start_transaction.return_value = mock_tx_context
    
    await mongo_repo.save_turn("conv1", [msg1, msg2])
    
    mock_db["messages"].insert_many.assert_called_once()
    mock_db["conversations"].update_one.assert_called_once()


@pytest.mark.asyncio
async def test_save_turn_failure_rolls_back(mongo_repo, mock_db):
    msg1 = StoredMessage("msg1", "conv1", "user", "hi", datetime.now(timezone.utc))
    
    mock_client = MagicMock()
    mock_db.client = mock_client
    
    mock_session = MagicMock()
    mock_client.start_session = AsyncMock(return_value=mock_session)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    
    mock_tx_context = MagicMock()
    mock_tx_context.__aenter__ = AsyncMock(return_value=None)
    mock_tx_context.__aexit__ = AsyncMock(return_value=None)
    mock_session.start_transaction.return_value = mock_tx_context
    
    mock_db["messages"].insert_many.side_effect = PyMongoError("Tx failed")
    
    with pytest.raises(MemoryStorageError, match="Failed to atomically save turn: Tx failed"):
        await mongo_repo.save_turn("conv1", [msg1])


class MockCursor:
    def __init__(self, items):
        self.items = items
    def sort(self, *args, **kwargs):
        return self
    async def __aiter__(self):
        for item in self.items:
            yield item

@pytest.mark.asyncio
async def test_get_messages_ordering(mongo_repo, mock_db):
    docs = [
        {"_id": "msg1", "conversation_id": "conv1", "role": "user", "content": "1", "created_at": datetime.now(timezone.utc), "metadata": {}},
        {"_id": "msg2", "conversation_id": "conv1", "role": "assistant", "content": "2", "created_at": datetime.now(timezone.utc), "metadata": {}},
    ]
    mock_db["messages"].find.return_value = MockCursor(docs)
    
    msgs = await mongo_repo.get_messages("conv1")
    
    mock_db["messages"].find.assert_called_with({"conversation_id": "conv1"})
    assert len(msgs) == 2
