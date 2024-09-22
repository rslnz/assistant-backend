import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Tool(BaseModel):
    name: str
    depends_on: List[str] = []

class Step(BaseModel):
    description: str
    status: str
    tools: List[Tool] = []

class Plan(BaseModel):
    steps: List[Step]
    current_step: int
    total_steps: int

class Reasoning(BaseModel):
    thought: str
    user_notification: str

class ToolUse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    arguments: Dict[str, Any]
    user_notification: str

class StatusEnum(str, Enum):
    CONTINUE = "continue"
    CLARIFY = "clarify"
    COMPLETE = "complete"

class Status(BaseModel):
    status: StatusEnum
    reason: Optional[str] = None