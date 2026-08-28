from typing import Literal
from pydantic import BaseModel, Field

DecisionAction = Literal["CREATE", "UPDATE", "DELETE", "IGNORE", "ASK_USER"]
MemorySourceEnum = Literal["explicit_user", "model_suggested", "system"]
MemoryTypeEnum = Literal["preference", "fact", "instruction", "core"]

class MemoryDecisionSchema(BaseModel):
    action: DecisionAction
    source: MemorySourceEnum
    rationale: str
    # Enforce strict 0.0 to 1.0 confidence as per V1.3 requirements
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Required for CREATE, optional otherwise
    memory_type: MemoryTypeEnum | None = None
    key: str | None = None
    content: str | None = None
    
    # Required for UPDATE/DELETE, optional otherwise
    target_memory_id: str | None = None
