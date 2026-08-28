"""
app/memory/exceptions.py

Exceptions raised by the memory layer.
"""

class MemoryError(Exception):
    """Base class for all memory-related errors."""
    pass


class ConversationNotFoundError(MemoryError):
    """Raised when a conversation cannot be found."""
    pass


class PersonalMemoryNotFoundError(MemoryError):
    """Raised when a personal memory cannot be found by ID."""
    pass


class MemoryStorageError(MemoryError):
    """Raised when there is a failure to read/write from the underlying storage."""
    pass
