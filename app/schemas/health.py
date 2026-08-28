"""
app/schemas/health.py — Health-check response schema.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
