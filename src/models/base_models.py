from enum import Enum
from typing import List
from pydantic import BaseModel, Field

class Role(str, Enum):
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"

class MessageEntry(BaseModel):
    role: Role
    content: str

class BaseContext(BaseModel):
    history: List[MessageEntry] = Field(default_factory=list)
    summary: str = ""