"""
app/logging_config.py

Configures structured logging for the application.

Request IDs are propagated through a contextvars.ContextVar so that every
log record emitted during a request automatically carries the same ID —
without passing it through every function call.
"""

import logging
import contextvars
from typing import Optional

# Module-level context variable; set by RequestIDMiddleware at request start.
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


class RequestIDFilter(logging.Filter):
    """Inject the current request ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("—")  # type: ignore[attr-defined]
        return True


def configure_logging(log_level: str = "INFO") -> None:
    """
    Set up root logger with a format that includes request ID.
    Call once at application startup.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    handler = logging.StreamHandler()
    handler.addFilter(RequestIDFilter())
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | req=%(request_id)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    # Remove any handlers already attached (e.g., uvicorn's default handlers)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
