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
async def test_save_message_success(mongo_repo, mock_db):
    msg = StoredMessage(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        content="hello",
        created_at=datetime.now(timezone.utc)
    )
    
    await mongo_repo.save_message(msg)
    mock_db["messages"].insert_one.assert_called_once()
    assert mock_db["messages"].create_index.assert_called
