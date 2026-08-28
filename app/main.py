"""
app/main.py — FastAPI application factory.

Creates and configures the FastAPI app:
  1. Logging initialised at startup.
  2. RequestIDMiddleware registered so every request gets a correlation ID.
  3. API router mounted.

Uvicorn entry point: `uvicorn app.main:app`
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.api.router import router as api_router
from app.config import get_settings
from app.logging_config import configure_logging
from app.middleware.request_id import RequestIDMiddleware
from app.llm.exceptions import LLMConfigurationError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    Eagerly initialises the LLM provider at startup so that configuration
    errors (e.g. missing API key) surface immediately with a clear log
    message rather than as a 500 on the first chat request.
    """
    from app.dependencies import get_llm_provider, get_mongo_client, get_conversation_repository, get_personal_memory_repository  # noqa: PLC0415

    try:
        get_llm_provider()
        logger.info("LLM provider initialised successfully")
    except LLMConfigurationError as exc:
        logger.error(
            "LLM configuration error — check your .env file: %s", exc
        )
        # We log and continue; the app will still serve /health and return
        # a clean error on /chat rather than crashing uvicorn.

    mongo_client = get_mongo_client()
    try:
        await mongo_client.connect()
        # Ensure indexes are created at startup
        conv_repo = get_conversation_repository()
        await conv_repo.initialize()
        
        pm_repo = get_personal_memory_repository()
        await pm_repo.initialize()
    except Exception as exc:
        # Do not log the exception string directly if it might contain the URI.
        # PyMongo errors generally don't leak the URI, but we log safely.
        logger.critical("CRITICAL: Failed to connect to MongoDB or initialize repository during startup.")
        # We MUST raise to prevent the application from starting in a degraded memory state.
        raise RuntimeError("MongoDB initialization failed") from exc

    yield  # Application runs here

    await mongo_client.close()


def create_app() -> FastAPI:
    settings = get_settings()

    configure_logging(settings.log_level)

    app = FastAPI(
        title="Mili",
        description="Mili — Agentic AI personal assistant (Genesis v0.1)",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Middleware (added in reverse order — outermost first in declaration order)
    app.add_middleware(RequestIDMiddleware)

    # Routes
    app.include_router(api_router)

    logger.info(
        "Mili %s starting (env=%s, provider=%s)",
        settings.app_version,
        settings.app_env,
        settings.llm_provider,
    )

    return app


app = create_app()
