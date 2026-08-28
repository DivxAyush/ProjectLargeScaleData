from dataclasses import dataclass
from datetime import datetime
from typing import Literal

# Defined sources
MemorySource = Literal["explicit_user", "model_suggested", "system"]
# Defined types
MemoryType = Literal["preference", "fact", "instruction", "core"]

@dataclass
class PersonalMemory:
    memory_id: str
    user_id: str
    memory_type: MemoryType
    key: str
    content: str
    source: MemorySource
    confidence: float
    created_at: datetime
    updated_at: datetime

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        valid_sources = {"explicit_user", "model_suggested", "system"}
        if self.source not in valid_sources:
            raise ValueError(f"Invalid source: {self.source}")
        
        valid_types = {"preference", "fact", "instruction", "core"}
        if self.memory_type not in valid_types:
            raise ValueError(f"Invalid memory type: {self.memory_type}")
