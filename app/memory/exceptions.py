"""
app/memory/exceptions.py

Exceptions raised by the memory layer.
"""

class MemoryError(Exception):
    """Base class for all memory-related errors."""
    pass


class ConversationNotFoundError(MemoryError):
    """Raised when a requested conversation ID does not exist."""
    pass


class MemoryStorageError(MemoryError):
    """Raised when there is a failure to read/write from the underlying storage."""
    pass
