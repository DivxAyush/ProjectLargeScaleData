"""
app/llm/providers/gemini.py

Google Gemini provider implementation.

ISOLATION BOUNDARY: This is the ONLY file in the codebase that imports
`google.genai`. No other module should import from this file directly;
all access goes through the LLMProvider Protocol.

SDK: google-genai (the unified, official SDK introduced to replace the
deprecated google-generativeai package, which reached EOL on 2025-08-31).
See: https://googleapis.github.io/python-genai/
"""

import logging
from typing import TYPE_CHECKING

from app.llm.base import Message
from app.llm.exceptions import LLMConfigurationError, LLMProviderError

if TYPE_CHECKING:
    # Only imported for type-checking; at runtime the import below handles it.
    from app.config import Settings

logger = logging.getLogger(__name__)


class GeminiProvider:
    """
    Concrete LLMProvider implementation backed by the Google Gemini API.

    Satisfies the LLMProvider Protocol structurally — no inheritance needed.

    Message format translation:
        Mili internal  →  Gemini API
        role="user"    →  role="user"
        role="assistant" → role="model"  (Gemini uses "model", not "assistant")
        role="system"  →  Prepended as a system_instruction (Gemini 1.5+ / 2.x)
    """

    def __init__(self, settings: "Settings") -> None:
        # google.genai import is scoped to this file only.
        from google import genai  # noqa: PLC0415

        if not settings.gemini_api_key:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example to .env and provide your API key."
            )

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model
        logger.info("GeminiProvider initialised (model=%s)", self._model)

    async def chat(self, messages: list[Message]) -> str:
        """
        Send the conversation history to Gemini and return the reply text.

        Raises:
            LLMProviderError: On any Gemini API or network failure.
        """
        from google.genai import types as genai_types  # noqa: PLC0415

        # Separate the optional system message from the conversation turns.
        system_instruction: str | None = None
        conversation: list[genai_types.ContentDict] = []

        for msg in messages:
            if msg.role == "system":
                # Gemini accepts a single system instruction at the top level.
                # If multiple system messages are present, we concatenate them.
                system_instruction = (
                    msg.content
                    if system_instruction is None
                    else f"{system_instruction}\n{msg.content}"
                )
            else:
                # Gemini uses "model" for assistant turns, "user" for user turns.
                gemini_role = "model" if msg.role == "assistant" else "user"
                conversation.append(
                    genai_types.ContentDict(
                        role=gemini_role,
                        parts=[genai_types.PartDict(text=msg.content)],
                    )
                )

        try:
            config = genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
            )
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=conversation,
                config=config,
            )
            return response.text or ""

        except Exception as exc:
            # Wrap SDK-specific exception so it never leaks upward.
            logger.error("Gemini API call failed: %s", exc)
            raise LLMProviderError(f"Gemini request failed: {exc}") from exc
