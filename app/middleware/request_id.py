"""
app/middleware/request_id.py

RequestID middleware — assigns a unique correlation ID to every request.

Behaviour:
  1. If the incoming request carries an X-Request-ID header, that value
     is used (allowing clients/gateways to propagate trace IDs).
  2. Otherwise, a new UUID4 is generated.
  3. The ID is stored in a contextvars.ContextVar so that every log line
     emitted during the request lifecycle carries it automatically.
  4. The ID is returned as an X-Request-ID response header.
  5. The ID is also available to route handlers via request.state.request_id,
     so it can be included in response bodies (e.g. ChatResponse.request_id).

This is the foundation for distributed tracing of agent/tool/RAG calls
in future versions.
"""

import uuid
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.logging_config import request_id_var

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # Honour a client-supplied ID or generate a fresh one.
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())

        # Store in context so logging filter can attach it to every log record.
        token = request_id_var.set(request_id)

        # Expose to route handlers via request.state.
        request.state.request_id = request_id

        logger.debug("Request started: %s %s", request.method, request.url.path)

        try:
            response: Response = await call_next(request)
        finally:
            # Always reset the context variable to avoid bleed between requests
            # in async frameworks that reuse threads/tasks.
            request_id_var.reset(token)

        response.headers[REQUEST_ID_HEADER] = request_id
        return response
