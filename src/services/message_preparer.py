from typing import List

from langchain.schema import (AIMessage, BaseMessage, HumanMessage,
                              SystemMessage)

from src.config import settings
from src.models.chat_models import ConversationContext


class MessagePreparer:
    def prepare_messages(self, context: ConversationContext, system_prompt: str, user_input: str) -> List[BaseMessage]:
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        if context.summary:
            messages.append(SystemMessage(content=f"Previous conversation summary: {context.summary}"))

        messages.extend(self._get_recent_messages(context))
        messages.append(HumanMessage(content=user_input))

        return messages

    def _get_recent_messages(self, context: ConversationContext) -> List[BaseMessage]:
        return [
            AIMessage(content=msg["content"]) if msg["role"] == "ai" else HumanMessage(content=msg["content"])
            for msg in context.history[-settings.MAX_HISTORY_MESSAGES:]
        ]
