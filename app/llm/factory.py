"""
app/llm/factory.py

Startup-only provider factory.

Responsibility: map the `llm_provider` config string to a concrete
LLMProvider instance. This is the ONLY place where provider class names
are referenced. Adding a new provider means adding one `case` here and
one file under app/llm/providers/.

This module should only be called from app/dependencies.py (the
composition root). Routes and services must never import this directly.
"""

import logging

from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.exceptions import LLMConfigurationError

logger = logging.getLogger(__name__)


def create_llm_provider(settings: Settings) -> LLMProvider:
    """
    Instantiate and return the configured LLM provider.

    Args:
        settings: Application settings (reads settings.llm_provider).

    Returns:
        A concrete object satisfying the LLMProvider Protocol.

    Raises:
        LLMConfigurationError: If the provider name is unknown.
    """
    provider_name = settings.llm_provider.lower()
    logger.info("Creating LLM provider: %s", provider_name)

    match provider_name:
        case "gemini":
            # Import is deferred to here so the SDK is only loaded when
            # the Gemini provider is actually selected.
            from app.llm.providers.gemini import GeminiProvider  # noqa: PLC0415

            return GeminiProvider(settings)

        # ── Future providers will be added here ──────────────────────────────
        # case "openai":
        #     from app.llm.providers.openai import OpenAIProvider
        #     return OpenAIProvider(settings)
        #
        # case "claude":
        #     from app.llm.providers.claude import ClaudeProvider
        #     return ClaudeProvider(settings)
        #
        # case "local":
        #     from app.llm.providers.local import LocalProvider
        #     return LocalProvider(settings)

        case _:
            raise LLMConfigurationError(
                f"Unknown LLM provider: '{provider_name}'. "
                f"Set LLM_PROVIDER in your .env to a supported value (e.g. 'gemini')."
            )
