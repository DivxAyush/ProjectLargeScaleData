import json
import pytest
from unittest.mock import AsyncMock, patch

from app.memory.manager import MemoryManager, CommandDetector
from app.memory.personal_models import PersonalMemory
from datetime import datetime, timezone

class FakeLLMResult:
    def __init__(self, reply: str):
        self.reply = reply

class FakeLLMProvider:
    def __init__(self):
        self.mock_reply = "{}"
    async def chat(self, messages):
        return FakeLLMResult(self.mock_reply)

@pytest.fixture
def fake_llm():
    return FakeLLMProvider()

@pytest.fixture
def fake_pm_service():
    service = AsyncMock()
    service.get_memories.return_value = []
    return service

@pytest.fixture
def manager(fake_llm, fake_pm_service):
    return MemoryManager(llm_provider=fake_llm, personal_memory_service=fake_pm_service)


def test_command_detector():
    detector = CommandDetector()
    assert detector.detect_intent("Mili, yaad rakhna I like pizza") == "explicit_user"
    assert detector.detect_intent("Please bhool jao my preference") == "explicit_user"
    assert detector.detect_intent("change this memory now") == "explicit_user"
    assert detector.detect_intent("I just want pizza") == "model_suggested"
    assert detector.detect_intent("What is the weather?") == "model_suggested"


@pytest.mark.asyncio
async def test_manager_create_explicit(manager, fake_llm, fake_pm_service):
    fake_llm.mock_reply = json.dumps({
        "action": "CREATE",
        "source": "explicit_user",
        "rationale": "Test",
        "confidence": 1.0,
        "memory_type": "preference",
        "key": "food",
        "content": "pizza"
    })
    # Must use explicit string to trigger explicit_user from detector
    await manager.evaluate_turn("u1", "Mili, yaad rakhna pizza")
    
    fake_pm_service.create_memory.assert_called_once_with(
        user_id="u1",
        memory_type="preference",
        key="food",
        content="pizza",
        source="explicit_user",
        confidence=1.0
    )


@pytest.mark.asyncio
async def test_manager_rejects_model_suggested(manager, fake_llm, fake_pm_service):
    # Even if LLM lies and says explicit_user, the detector will override it to model_suggested
    fake_llm.mock_reply = json.dumps({
        "action": "CREATE",
        "source": "explicit_user",
        "rationale": "Test",
        "confidence": 1.0,
        "memory_type": "preference",
        "key": "food",
        "content": "pizza"
    })
    # Implicit message
    await manager.evaluate_turn("u1", "I like pizza")
    
    fake_pm_service.create_memory.assert_not_called()


@pytest.mark.asyncio
async def test_manager_update_ownership(manager, fake_llm, fake_pm_service):
    # Setup existing memory owned by u1
    fake_pm_service.get_memories.return_value = [
        PersonalMemory(
            memory_id="mem123", user_id="u1", memory_type="preference", 
            key="food", content="burger", source="explicit_user", confidence=1.0, 
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
        )
    ]
    
    fake_llm.mock_reply = json.dumps({
        "action": "UPDATE",
        "source": "explicit_user",
        "rationale": "Change",
        "confidence": 1.0,
        "content": "pizza",
        "target_memory_id": "mem123"
    })
    
    await manager.evaluate_turn("u1", "Mili, change this memory to pizza")
    
    fake_pm_service.update_memory.assert_called_once_with(
        memory_id="mem123",
        content="pizza",
        confidence=1.0
    )

@pytest.mark.asyncio
async def test_manager_update_unowned_target_is_rejected(manager, fake_llm, fake_pm_service):
    # Memory does not belong to u1 (not returned by get_memories)
    fake_pm_service.get_memories.return_value = []
    
    fake_llm.mock_reply = json.dumps({
        "action": "UPDATE",
        "source": "explicit_user",
        "rationale": "Change",
        "confidence": 1.0,
        "content": "pizza",
        "target_memory_id": "mem123"
    })
    
    await manager.evaluate_turn("u1", "Mili, change this memory to pizza")
    
    fake_pm_service.update_memory.assert_not_called()

@pytest.mark.asyncio
async def test_manager_delete_ownership(manager, fake_llm, fake_pm_service):
    # Setup existing memory owned by u1
    fake_pm_service.get_memories.return_value = [
        PersonalMemory(
            memory_id="mem123", user_id="u1", memory_type="preference", 
            key="food", content="burger", source="explicit_user", confidence=1.0, 
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
        )
    ]
    
    fake_llm.mock_reply = json.dumps({
        "action": "DELETE",
        "source": "explicit_user",
        "rationale": "Remove",
        "confidence": 1.0,
        "target_memory_id": "mem123"
    })
    
    await manager.evaluate_turn("u1", "Mili, bhool jao mem123")
    
    fake_pm_service.delete_memory.assert_called_once_with(memory_id="mem123")


@pytest.mark.asyncio
async def test_manager_ignores_ignore_and_ask_user(manager, fake_llm, fake_pm_service):
    fake_llm.mock_reply = json.dumps({
        "action": "IGNORE",
        "source": "explicit_user",
        "rationale": "temp",
        "confidence": 1.0
    })
    await manager.evaluate_turn("u1", "Mili, yaad rakhna temporary stuff")
    fake_pm_service.create_memory.assert_not_called()

    fake_llm.mock_reply = json.dumps({
        "action": "ASK_USER",
        "source": "explicit_user",
        "rationale": "conflict",
        "confidence": 1.0
    })
    await manager.evaluate_turn("u1", "Mili, yaad rakhna conflict")
    fake_pm_service.create_memory.assert_not_called()

@pytest.mark.asyncio
async def test_manager_handles_invalid_json(manager, fake_llm, fake_pm_service):
    fake_llm.mock_reply = "I am not JSON"
    await manager.evaluate_turn("u1", "Mili, yaad rakhna")
    fake_pm_service.create_memory.assert_not_called()

@pytest.mark.asyncio
async def test_manager_handles_schema_failure(manager, fake_llm, fake_pm_service):
    fake_llm.mock_reply = json.dumps({"action": "INVALID", "confidence": 99.0})
    await manager.evaluate_turn("u1", "Mili, yaad rakhna")
    fake_pm_service.create_memory.assert_not_called()
