from typing import Optional
from pydantic import BaseModel, Field
from src.models.base_models import BaseContext

class ConversationContext(BaseContext):
    pass

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    system_prompt: str = Field(default="", description="System prompt for the assistant")
    context: Optional[ConversationContext] = Field(default=None, description="Conversation context from the previous response")

class ChatResponse(BaseModel):
    ai_message: str
    updated_context: ConversationContext