import pytest
from datetime import datetime, timezone
from app.memory.personal_models import PersonalMemory

def test_personal_memory_valid():
    pm = PersonalMemory(
        memory_id="123",
        user_id="u1",
        memory_type="preference",
        key="diet",
        content="vegan",
        source="explicit_user",
        confidence=0.9,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    assert pm.confidence == 0.9

def test_personal_memory_invalid_confidence():
    with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
        PersonalMemory(
            memory_id="123",
            user_id="u1",
            memory_type="preference",
            key="diet",
            content="vegan",
            source="explicit_user",
            confidence=1.5,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

def test_personal_memory_invalid_source():
    with pytest.raises(ValueError, match="Invalid source: magical"):
        PersonalMemory(
            memory_id="123",
            user_id="u1",
            memory_type="preference",
            key="diet",
            content="vegan",
            source="magical", # type: ignore
            confidence=1.0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

def test_personal_memory_invalid_type():
    with pytest.raises(ValueError, match="Invalid memory type: dream"):
        PersonalMemory(
            memory_id="123",
            user_id="u1",
            memory_type="dream", # type: ignore
            key="diet",
            content="vegan",
            source="explicit_user",
            confidence=1.0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
