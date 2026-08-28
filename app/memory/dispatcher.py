from typing import Protocol
from fastapi import BackgroundTasks

class MemoryTaskDispatcher(Protocol):
    """
    Decouples background memory evaluation from the web framework.
    """
    def dispatch(self, user_id: str, message: str) -> None:
        ...

class FastAPIMemoryDispatcher(MemoryTaskDispatcher):
    """
    Implementation using FastAPI's native BackgroundTasks.
    """
    def __init__(self, background_tasks: BackgroundTasks, memory_manager: "MemoryManager") -> None: # type: ignore
        self._background_tasks = background_tasks
        self._memory_manager = memory_manager

    def dispatch(self, user_id: str, message: str) -> None:
        # Enqueue the evaluate_turn coroutine to run after the HTTP response
        self._background_tasks.add_task(self._memory_manager.evaluate_turn, user_id, message)
