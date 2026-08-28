import pytest
from pydantic import ValidationError
from app.memory.manager_schemas import MemoryDecisionSchema

def test_valid_memory_decision():
    schema = MemoryDecisionSchema(
        action="CREATE",
        source="explicit_user",
        rationale="Because user said so",
        confidence=1.0,
        memory_type="preference",
        key="diet",
        content="vegan"
    )
    assert schema.confidence == 1.0

def test_invalid_confidence_above_1():
    with pytest.raises(ValidationError) as exc_info:
        MemoryDecisionSchema(
            action="CREATE",
            source="explicit_user",
            rationale="Because user said so",
            confidence=1.1,
        )
    assert "Input should be less than or equal to 1" in str(exc_info.value)

def test_invalid_confidence_below_0():
    with pytest.raises(ValidationError) as exc_info:
        MemoryDecisionSchema(
            action="CREATE",
            source="explicit_user",
            rationale="Because user said so",
            confidence=-0.1,
        )
    assert "Input should be greater than or equal to 0" in str(exc_info.value)

def test_invalid_action():
    with pytest.raises(ValidationError) as exc_info:
        MemoryDecisionSchema(
            action="MAGICAL_ACTION", # type: ignore
            source="explicit_user",
            rationale="Because",
            confidence=1.0,
        )
    assert "Input should be 'CREATE', 'UPDATE', 'DELETE', 'IGNORE' or 'ASK_USER'" in str(exc_info.value)
