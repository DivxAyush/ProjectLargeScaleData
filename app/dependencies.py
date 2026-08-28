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
from app.llm.base import LLMProvider
from app.llm.factory import create_llm_provider
from app.services.chat_service import ChatService

# Module-level singleton — set on first successful initialisation.
_llm_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    """
    Create and cache the configured LLM provider.

    Only caches a successfully created provider. If provider creation fails
    (e.g. missing API key), the next call will try again rather than
    permanently returning the cached failure.
    """
    global _llm_provider
    if _llm_provider is None:
        settings = get_settings()
        _llm_provider = create_llm_provider(settings)
    return _llm_provider


def get_chat_service() -> ChatService:
    """
    FastAPI dependency that provides a ChatService instance.

    The provider is resolved once (cached); a new ChatService wrapper is
    created per request — lightweight since ChatService is stateless in v0.1.
    In future versions this may inject memory/context per-session.
    """
    provider = get_llm_provider()
    return ChatService(provider=provider)

