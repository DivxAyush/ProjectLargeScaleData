"""
app/config.py

Single source of truth for all configuration.
Values are read from environment variables (or a .env file).
Never put secrets in this file — only field definitions and defaults.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    app_port: int = 8000
    app_version: str = "0.1.0"

    # ── LLM provider selection ───────────────────────────────────────────────
    # Changing this value (plus providing the corresponding API key) is all
    # that is needed to switch providers. The rest of the application is
    # provider-agnostic.
    llm_provider: str = "gemini"

    # ── Gemini-specific ──────────────────────────────────────────────────────
    # These fields are only read by app/llm/providers/gemini.py.
    # No other module should access them.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.7-flash"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()
