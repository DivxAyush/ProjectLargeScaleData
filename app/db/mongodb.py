"""
app/db/mongodb.py

MongoDB infrastructure module using the official PyMongo Async API.
Responsible ONLY for client connection lifecycle.
"""

import logging

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


class MongoDBClient:
    """
    Manages the lifecycle of the PyMongo AsyncMongoClient.
    """

    def __init__(self, uri: str, db_name: str) -> None:
        self._uri = uri
        self._db_name = db_name
        self._client: AsyncMongoClient | None = None

    async def connect(self) -> None:
        """Initialize the client. Should be called during app startup."""
        if self._client is not None:
            return

        logger.info("Connecting to MongoDB (DB: %s)", self._db_name)
        try:
            # We delay passing the URI to the client until connect() is called.
            # In a real app we might pass standard connection pooling settings here.
            self._client = AsyncMongoClient(self._uri, uuidRepresentation="standard")
            
            # Verify connection immediately.
            await self._client.admin.command('ping')
            logger.info("Successfully connected to MongoDB.")
        except PyMongoError as exc:
            logger.error("Failed to connect to MongoDB: %s", exc)
            self._client = None
            raise

    async def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client:
            logger.info("Closing MongoDB connection.")
            await self._client.close()
            self._client = None

    def get_database(self) -> AsyncDatabase:
        """
        Return the AsyncDatabase instance.
        Raises RuntimeError if not connected.
        """
        if self._client is None:
            raise RuntimeError("MongoDB client is not connected.")
        return self._client[self._db_name]

    async def ping(self) -> bool:
        """Health check utility."""
        if self._client is None:
            return False
        try:
            await self._client.admin.command('ping')
            return True
        except PyMongoError:
            return False
