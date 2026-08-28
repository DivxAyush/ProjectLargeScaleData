"""
app/api/router.py — Top-level API router.

Mounts all sub-routers under the /api prefix.
"""

from fastapi import APIRouter

from app.api import health, chat

router = APIRouter(prefix="/api")
router.include_router(health.router, tags=["health"])
router.include_router(chat.router, tags=["chat"])
