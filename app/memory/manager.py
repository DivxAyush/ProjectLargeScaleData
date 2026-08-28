import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from app.llm.base import LLMProvider, Message
from app.memory.manager_schemas import MemoryDecisionSchema, MemorySourceEnum
from app.memory.personal_service import PersonalMemoryService

logger = logging.getLogger(__name__)

class CommandDetector:
    """
    Deterministically detects explicit user memory commands before LLM inference.
    """
    def __init__(self) -> None:
        self.explicit_patterns = [
            r"(?i)\byaad rakhna\b",
            r"(?i)\bbhool jao\b",
            r"(?i)\bchange this memory\b",
            r"(?i)\bremember this\b",
            r"(?i)\bforget this\b"
        ]

    def detect_intent(self, message: str) -> MemorySourceEnum:
        """
        Returns 'explicit_user' if an explicit command is detected,
        otherwise returns 'model_suggested'.
        """
        for pattern in self.explicit_patterns:
            if re.search(pattern, message):
                return "explicit_user"
        return "model_suggested"


class MemoryManager:
    """
    Intelligent Memory Manager for evaluating conversation turns and extracting
    long-term personal memory. Applies strict business rules to LLM outputs.
    """
    def __init__(self, llm_provider: LLMProvider, personal_memory_service: PersonalMemoryService) -> None:
        self._provider = llm_provider
        self._pm_service = personal_memory_service
        self._detector = CommandDetector()

    async def evaluate_turn(self, user_id: str, message: str) -> None:
        """
        Evaluate the turn asynchronously and mutate memory if strict criteria are met.
        """
        # 1. Deterministic intent detection
        intent_source = self._detector.detect_intent(message)

        # 2. Fetch existing memories for contradiction check
        existing_memories = await self._pm_service.get_memories(user_id=user_id, limit=50)

        # 3. Format prompt and call LLM
        prompt = self._build_prompt(intent_source, message, existing_memories)
        
        try:
            llm_result = await self._provider.chat([Message(role="user", content=prompt)])
            raw_output = llm_result.reply
        except Exception as e:
            logger.error("LLM failure during memory evaluation: %s", type(e).__name__)
            return

        # 4. JSON parsing and Pydantic validation
        decision = self._parse_and_validate(raw_output)
        if not decision:
            return  # Invalid decision -> no mutation, safely halts

        # 5. Apply business safety rules & User isolation
        await self._execute_decision(user_id, decision, intent_source, existing_memories)

    def _build_prompt(self, intent_source: MemorySourceEnum, message: str, existing_memories: list) -> str:
        # In a real app, this would be a well-crafted prompt template.
        memories_str = "\n".join(
            [f"ID: {m.memory_id} | Type: {m.memory_type} | Content: {m.content}" for m in existing_memories]
        )
        return f"""
        Analyze the following message for long-term personal memory extraction.
        Intent detected by system: {intent_source}
        Existing Memories:
        {memories_str if memories_str else "None"}
        
        Message: "{message}"
        
        Respond ONLY with a valid JSON object matching the MemoryDecisionSchema:
        {{
            "action": "CREATE|UPDATE|DELETE|IGNORE|ASK_USER",
            "source": "{intent_source}",
            "rationale": "Why...",
            "confidence": 0.0-1.0,
            "memory_type": "preference|fact|instruction|core",
            "key": "...",
            "content": "...",
            "target_memory_id": "..."
        }}
        """

    def _parse_and_validate(self, raw_output: str) -> MemoryDecisionSchema | None:
        try:
            # Strip potential markdown blocks (e.g. ```json ... ```)
            clean_output = raw_output.strip()
            if clean_output.startswith("```json"):
                clean_output = clean_output[7:]
            if clean_output.startswith("```"):
                clean_output = clean_output[3:]
            if clean_output.endswith("```"):
                clean_output = clean_output[:-3]
            
            parsed_json = json.loads(clean_output.strip())
            return MemoryDecisionSchema(**parsed_json)
        except json.JSONDecodeError as e:
            logger.warning("MemoryDecision validation failed: json_decode_error")
            return None
        except ValidationError as e:
            logger.warning("MemoryDecision validation failed: invalid_schema")
            return None

    async def _execute_decision(
        self, user_id: str, decision: MemoryDecisionSchema, intent_source: MemorySourceEnum, existing_memories: list
    ) -> None:
        # Business Rule 1: Ignore/Ask_User = no mutation
        if decision.action in ("IGNORE", "ASK_USER"):
            return

        # Business Rule 2: Override source strictly with our deterministic intent detector
        # The LLM shouldn't be able to lie and claim explicit_user if we didn't detect it.
        actual_source = intent_source
        
        # Business Rule 3: Rejection of model_suggested and system sources in V1.3
        if actual_source != "explicit_user":
            logger.info("Dropping memory decision due to source=%s", actual_source)
            return

        # Execute actions
        try:
            if decision.action == "CREATE":
                if not decision.memory_type or not decision.key or not decision.content:
                    logger.warning("CREATE decision missing required fields.")
                    return
                await self._pm_service.create_memory(
                    user_id=user_id,
                    memory_type=decision.memory_type, # type: ignore
                    key=decision.key,
                    content=decision.content,
                    source="explicit_user",
                    confidence=decision.confidence,
                )
            elif decision.action == "UPDATE":
                if not decision.target_memory_id or not decision.content:
                    logger.warning("UPDATE decision missing target_memory_id or content.")
                    return
                # User isolation / Ownership check
                if not any(m.memory_id == decision.target_memory_id for m in existing_memories):
                    logger.warning("UPDATE target_memory_id %s not found or not owned by user %s", decision.target_memory_id, user_id)
                    return
                await self._pm_service.update_memory(
                    memory_id=decision.target_memory_id,
                    content=decision.content,
                    confidence=decision.confidence,
                )
            elif decision.action == "DELETE":
                if not decision.target_memory_id:
                    logger.warning("DELETE decision missing target_memory_id.")
                    return
                # User isolation / Ownership check
                if not any(m.memory_id == decision.target_memory_id for m in existing_memories):
                    logger.warning("DELETE target_memory_id %s not found or not owned by user %s", decision.target_memory_id, user_id)
                    return
                await self._pm_service.delete_memory(memory_id=decision.target_memory_id)
        except Exception as e:
            logger.error("Failed to execute memory decision: %s", type(e).__name__)
