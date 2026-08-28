"""
tests/test_chat.py — Chat endpoint HTTP tests.

These tests verify:
  - Correct response schema
  - Request ID propagation in body and header
  - Input validation (empty messages)
  - Error mapping: LLMProviderError → HTTP 502
"""

import pytest
from httpx import AsyncClient

from tests.conftest import MockLLMProvider


@pytest.mark.asyncio
async def test_chat_returns_200(client: AsyncClient) -> None:
    response = await client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Hello!"}]},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_response_schema(client: AsyncClient) -> None:
    response = await client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Hello!"}]},
    )
    body = response.json()
    assert "reply" in body
    assert "request_id" in body
    assert body["reply"] == MockLLMProvider.FIXED_REPLY


@pytest.mark.asyncio
async def test_chat_request_id_in_body_matches_header(client: AsyncClient) -> None:
    response = await client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Hello!"}]},
    )
    body = response.json()
    assert body["request_id"] == response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_chat_empty_messages_returns_422(client: AsyncClient) -> None:
    """Empty messages list must fail Pydantic validation."""
    response = await client.post("/api/chat", json={"messages": []})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_missing_messages_returns_422(client: AsyncClient) -> None:
    response = await client.post("/api/chat", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_empty_content_returns_422(client: AsyncClient) -> None:
    """Empty string content must fail Pydantic validation (min_length=1)."""
    response = await client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": ""}]},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_invalid_role_returns_422(client: AsyncClient) -> None:
    response = await client.post(
        "/api/chat",
        json={"messages": [{"role": "unknown_role", "content": "Hi"}]},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_upstream_error_returns_502(failing_client: AsyncClient) -> None:
    """LLMProviderError from the provider must map to HTTP 502."""
    response = await failing_client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Hello!"}]},
    )
    assert response.status_code == 502
    body = response.json()
    assert body["detail"]["error"] == "upstream_llm_error"


@pytest.mark.asyncio
async def test_chat_502_includes_request_id(failing_client: AsyncClient) -> None:
    response = await failing_client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Hello!"}]},
    )
    assert response.status_code == 502
    body = response.json()
    assert "request_id" in body["detail"]


@pytest.mark.asyncio
async def test_chat_multi_turn_conversation(client: AsyncClient) -> None:
    """Multiple messages (multi-turn) should be accepted and processed."""
    response = await client.post(
        "/api/chat",
        json={
            "messages": [
                {"role": "system", "content": "You are Mili."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
        },
    )
    assert response.status_code == 200
