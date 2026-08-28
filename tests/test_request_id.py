"""
tests/test_request_id.py

Tests for RequestIDMiddleware — correlation ID generation and propagation.
"""

import uuid

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_response_contains_request_id_header(client: AsyncClient) -> None:
    response = await client.get("/api/health")
    assert "x-request-id" in response.headers


@pytest.mark.asyncio
async def test_auto_generated_request_id_is_valid_uuid(client: AsyncClient) -> None:
    response = await client.get("/api/health")
    request_id = response.headers["x-request-id"]
    # Must be parseable as a UUID — no ValueError means it's valid.
    parsed = uuid.UUID(request_id)
    assert parsed.version == 4


@pytest.mark.asyncio
async def test_client_supplied_request_id_is_preserved(client: AsyncClient) -> None:
    """If the client sends X-Request-ID, the same value must be echoed back."""
    custom_id = "my-trace-id-abc123"
    response = await client.get(
        "/api/health", headers={"X-Request-ID": custom_id}
    )
    assert response.headers["x-request-id"] == custom_id


@pytest.mark.asyncio
async def test_request_id_in_chat_response_body_matches_header(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    body = response.json()
    assert body["request_id"] == response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_each_request_gets_unique_id(client: AsyncClient) -> None:
    """Two sequential requests should have different auto-generated IDs."""
    r1 = await client.get("/api/health")
    r2 = await client.get("/api/health")
    assert r1.headers["x-request-id"] != r2.headers["x-request-id"]


@pytest.mark.asyncio
async def test_client_supplied_id_appears_in_chat_body(client: AsyncClient) -> None:
    custom_id = "trace-99887766"
    response = await client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Hi"}]},
        headers={"X-Request-ID": custom_id},
    )
    body = response.json()
    assert body["request_id"] == custom_id
