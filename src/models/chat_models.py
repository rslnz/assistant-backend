from enum import Enum
from typing import List, Optional

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

    def add_message(self, role: Role, content: str):
        self.history.append(MessageEntry(role=role, content=content))

    def needs_summarization(self) -> bool:
        return len(self.history) >= 20

    def get_messages_to_summarize(self) -> List[MessageEntry]:
        return self.history[:10]

    def remove_summarized_messages(self):
        self.history = self.history[10:]

class ConversationContext(BaseContext):
    pass

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    system_prompt: str = Field(default="", description="System prompt for the assistant")
    context: Optional[ConversationContext] = Field(default=None, description="Conversation context from the previous response")

class ChatResponse(BaseModel):
    ai_message: str
    updated_context: ConversationContext