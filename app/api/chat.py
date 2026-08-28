"""
app/api/chat.py — Chat endpoint.

ARCHITECTURE INVARIANT: This file imports only:
  - FastAPI primitives (APIRouter, Depends, HTTPException, Request)
  - ChatService (from app/services — the service layer)
  - ChatRequest, ChatResponse (from app/schemas — HTTP-boundary types)
  - LLMError (from app/llm/exceptions — provider-agnostic)
  - get_chat_service (from app/dependencies — composition root)

It must NOT import any LLM provider SDK, the factory, or concrete
provider classes. Route logic is intentionally thin.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.dependencies import get_chat_service
from app.llm.exceptions import LLMConfigurationError, LLMError
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with Mili",
    description=(
        "Send a conversation history and receive the assistant's reply. "
        "The request_id in the response body matches the X-Request-ID header."
    ),
)
async def chat(
    request: Request,
    body: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """
    POST /api/chat

    Accepts a list of messages and returns the assistant's reply.
    Maps upstream LLM failures to HTTP 502 so clients receive a clean error.
    """
    request_id: str = request.state.request_id

    logger.info("Chat request received (messages=%d)", len(body.messages))

    try:
        reply = await service.chat(body.messages)
    except LLMConfigurationError as exc:
        logger.error("LLM not configured: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "llm_not_configured",
                "message": str(exc),
                "request_id": request_id,
            },
        ) from exc
    except LLMError as exc:
        logger.error("LLM error during chat: %s", exc)
        raise HTTPException(
            status_code=502,
            detail={
                "error": "upstream_llm_error",
                "message": str(exc),
                "request_id": request_id,
            },
        ) from exc

    return ChatResponse(reply=reply, request_id=request_id)
