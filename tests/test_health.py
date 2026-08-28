"""
tests/test_health.py — Health endpoint tests.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient) -> None:
    response = await client.get("/api/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_schema(client: AsyncClient) -> None:
    response = await client.get("/api/health")
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert body["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_health_has_request_id_header(client: AsyncClient) -> None:
    response = await client.get("/api/health")
    assert "x-request-id" in response.headers
