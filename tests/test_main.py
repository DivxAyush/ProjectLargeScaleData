"""
tests/test_main.py

Tests for application startup lifecycle.
"""

import pytest
from unittest.mock import AsyncMock, patch
from app.main import app
from httpx import ASGITransport, AsyncClient

@pytest.mark.asyncio
async def test_startup_fails_if_mongo_fails():
    """Verify that a failure to connect to MongoDB aborts application startup."""
    with patch("app.db.mongodb.MongoDBClient.connect", new_callable=AsyncMock) as mock_connect:
        mock_connect.side_effect = Exception("Simulated connection failure")
        
        with pytest.raises(RuntimeError, match="MongoDB initialization failed"):
            async with app.router.lifespan_context(app):
                pass
