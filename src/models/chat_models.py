from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel

from src.config import settings


class Role(str, Enum):
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"

class MessageEntry(BaseModel):
    role: Role
    """Message role"""
    content: str
    """Message content"""

class ConversationContext(BaseModel):
    """Conversation context model"""
    history: List[MessageEntry] = []
    """Conversation history"""
    summary: str = ""
    """Latest summary of the conversation, including plan, reasoning, and status"""

    def add_message(self, role: Role, content: str):
        """Adds a message to the conversation history"""
        self.history.append(MessageEntry(role=role, content=content))

    def set_summary(self, summary: str):
        """Updates the conversation summary"""
        self.summary = summary

    def get_recent_messages(self, limit: int = settings.MAX_HISTORY_MESSAGES) -> List[Dict[str, str]]:
        """Returns the most recent messages from the conversation history"""
        return [entry.model_dump() for entry in self.history[-limit:]]

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    """User message"""
    system_prompt: str = ""
    """System prompt for the assistant"""
    context: Optional[ConversationContext] = None
    """Conversation context from the previous response"""
