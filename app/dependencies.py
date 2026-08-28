"""
app/dependencies.py

Composition root — FastAPI dependency providers.

This is the ONLY place that wires concrete implementations to abstractions.
Routes import from here; they never touch the factory or provider classes.

Design:
  - get_settings() is cached by pydantic-settings (lru_cache in config.py).
  - get_llm_provider() creates the provider once at startup and caches it.
    We use an explicit module-level singleton rather than @lru_cache because
    lru_cache also caches raised exceptions — an LLMConfigurationError on
    first call would permanently prevent recovery on subsequent calls.
  - get_chat_service() creates a ChatService per-request but shares the
    cached provider — this is efficient and keeps the service stateless.
"""

from app.config import get_settings
from app.db.mongodb import MongoDBClient
from app.llm.base import LLMProvider
from app.llm.factory import create_llm_provider
from app.memory.repository import ConversationRepository
from app.memory.repositories.mongo import MongoConversationRepository
from app.memory.service import MemoryService
from app.memory.personal_repository import PersonalMemoryRepository
from app.memory.personal_service import PersonalMemoryService
from app.memory.repositories.personal_mongo import MongoPersonalMemoryRepository
from app.services.chat_service import ChatService
from app.memory.manager import MemoryManager
from app.memory.dispatcher import MemoryTaskDispatcher, FastAPIMemoryDispatcher
from fastapi import BackgroundTasks

# Module-level singletons
_llm_provider: LLMProvider | None = None
_mongo_client: MongoDBClient | None = None
_conversation_repository: ConversationRepository | None = None
_memory_service: MemoryService | None = None
_personal_memory_repository: PersonalMemoryRepository | None = None
_personal_memory_service: PersonalMemoryService | None = None


def get_llm_provider() -> LLMProvider:
    """
    Create and cache the configured LLM provider.
    """
    global _llm_provider
    if _llm_provider is None:
        settings = get_settings()
        _llm_provider = create_llm_provider(settings)
    return _llm_provider


def get_mongo_client() -> MongoDBClient:
    """
    Create and cache the MongoDB client.
    """
    global _mongo_client
    if _mongo_client is None:
        settings = get_settings()
        _mongo_client = MongoDBClient(
            uri=settings.mongodb_uri,
            db_name=settings.mongodb_db_name,
        )
    return _mongo_client


def get_conversation_repository() -> ConversationRepository:
    """
    Create and cache the ConversationRepository.
    """
    global _conversation_repository
    if _conversation_repository is None:
        client = get_mongo_client()
        db = client.get_database()
        _conversation_repository = MongoConversationRepository(db)
    return _conversation_repository


def get_memory_service() -> MemoryService:
    """
    Create and cache the MemoryService.
    """
    global _memory_service
    if _memory_service is None:
        repo = get_conversation_repository()
        _memory_service = MemoryService(repository=repo)
    return _memory_service


def get_personal_memory_repository() -> PersonalMemoryRepository:
    """
    Create and cache the PersonalMemoryRepository.
    """
    global _personal_memory_repository
    if _personal_memory_repository is None:
        client = get_mongo_client()
        db = client.get_database()
        _personal_memory_repository = MongoPersonalMemoryRepository(db)
    return _personal_memory_repository


def get_personal_memory_service() -> PersonalMemoryService:
    """
    Create and cache the PersonalMemoryService.
    """
    global _personal_memory_service
    if _personal_memory_service is None:
        repo = get_personal_memory_repository()
        _personal_memory_service = PersonalMemoryService(repository=repo)
    return _personal_memory_service

_memory_manager: MemoryManager | None = None

def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        llm = get_llm_provider()
        pm_service = get_personal_memory_service()
        _memory_manager = MemoryManager(llm_provider=llm, personal_memory_service=pm_service)
    return _memory_manager

def get_memory_dispatcher(background_tasks: BackgroundTasks) -> MemoryTaskDispatcher:
    manager = get_memory_manager()
    return FastAPIMemoryDispatcher(background_tasks=background_tasks, memory_manager=manager)


def get_current_user_id() -> str:
    """
    FastAPI dependency that provides the current user's ID.
    In V1.2, this is hardcoded to 'default_user' as authentication is not yet built.
    """
    return "default_user"


def get_chat_service() -> ChatService:
    """
    FastAPI dependency that provides a ChatService instance.
    """
    provider = get_llm_provider()
    memory_service = get_memory_service()
    personal_memory_service = get_personal_memory_service()
    return ChatService(
        provider=provider, 
        memory_service=memory_service,
        personal_memory_service=personal_memory_service
    )
