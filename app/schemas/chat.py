"""
app/schemas/chat.py

Request and response schemas for the /api/chat endpoint.

These are HTTP-boundary models only. They are converted to/from the
internal Message model by ChatService — the API layer never touches
the internal representation directly.
"""

from typing import Literal

from pydantic import BaseModel, Field


class MessageSchema(BaseModel):
    """A single message in the conversation, as received/sent over HTTP."""

    role: Literal["user", "assistant", "system"] = Field(
        ...,
        description="The role of the message author.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The text content of the message.",
    )


class ChatRequest(BaseModel):
    """Request body for POST /api/chat."""

    messages: list[MessageSchema] = Field(
        ...,
        min_length=1,
        description="Ordered conversation history. Must contain at least one message.",
    )


class ChatResponse(BaseModel):
    """Response body for POST /api/chat."""

    reply: str = Field(..., description="The assistant's reply.")
    request_id: str = Field(
        ...,
        description="Correlation ID for this request. Matches the X-Request-ID header.",
    )
