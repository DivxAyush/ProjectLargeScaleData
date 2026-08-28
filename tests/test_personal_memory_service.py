import pytest
from app.memory.personal_service import PersonalMemoryService
from app.memory.exceptions import PersonalMemoryNotFoundError

@pytest.fixture
def pm_service(fake_personal_repo):
    return PersonalMemoryService(fake_personal_repo)

@pytest.mark.asyncio
async def test_create_and_get_memory(pm_service, fake_personal_repo):
    mem_id = await pm_service.create_memory(
        user_id="u1",
        memory_type="preference",
        key="diet",
        content="vegan",
        source="explicit_user",
    )
    assert mem_id in fake_personal_repo.memories
    
    memories = await pm_service.get_memories("u1")
    assert len(memories) == 1
    assert memories[0].memory_id == mem_id
    assert memories[0].content == "vegan"

@pytest.mark.asyncio
async def test_update_memory(pm_service, fake_personal_repo):
    mem_id = await pm_service.create_memory(
        user_id="u1",
        memory_type="preference",
        key="diet",
        content="vegan",
        source="explicit_user",
    )
    
    await pm_service.update_memory(mem_id, content="no dairy", confidence=0.8)
    
    memories = await pm_service.get_memories("u1")
    assert len(memories) == 1
    assert memories[0].content == "no dairy"
    assert memories[0].confidence == 0.8

@pytest.mark.asyncio
async def test_delete_memory(pm_service, fake_personal_repo):
    mem_id = await pm_service.create_memory(
        user_id="u1",
        memory_type="preference",
        key="diet",
        content="vegan",
        source="explicit_user",
    )
    
    await pm_service.delete_memory(mem_id)
    
    memories = await pm_service.get_memories("u1")
    assert len(memories) == 0

@pytest.mark.asyncio
async def test_update_nonexistent_memory(pm_service):
    with pytest.raises(PersonalMemoryNotFoundError):
        await pm_service.update_memory("missing", "content")

@pytest.mark.asyncio
async def test_delete_nonexistent_memory(pm_service):
    with pytest.raises(PersonalMemoryNotFoundError):
        await pm_service.delete_memory("missing")
