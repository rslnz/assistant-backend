from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum

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

class ClientCommand(BaseModel):
    name: str
    """Command name"""
    description: str
    """Command description"""
    arguments: Dict[str, str]
    """Command arguments, where key is argument name and value is description"""

class ConversationContext(BaseModel):
    """Conversation context model"""
    conversation_history: List[MessageEntry] = []
    """Conversation history"""
    conversation_summary: Optional[str] = None
    """Brief summary of previous messages"""

    def add_message(self, role: Role, content: str):
        """Adds a message to the conversation history"""
        self.conversation_history.append(MessageEntry(role=role, content=content))

    def set_summary(self, summary: str):
        """Updates the conversation summary"""
        self.conversation_summary = summary

    def get_recent_messages(self, limit: int = settings.MAX_HISTORY_MESSAGES) -> List[Dict[str, str]]:
        """Returns the most recent messages from the conversation history"""
        return [entry.model_dump() for entry in self.conversation_history[-limit:]]

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    """User message"""
    system_prompt: Optional[str] = None
    """System prompt for the assistant"""
    context: Optional[ConversationContext] = None
    """Conversation context from the previous response"""
