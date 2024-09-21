import json
import logging
from typing import Any, AsyncGenerator, Dict

from src.models.chat_models import ChatRequest, ConversationContext
from src.services.conversation_agent import ConversationAgent
from src.services.llm_service import LLMService

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self, conversation_agent: ConversationAgent):
        self.llm_service = LLMService()
        self.conversation_agent = conversation_agent

    async def process_chat_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        message = request.message
        system_prompt = request.system_prompt
        context = request.context or ConversationContext()

        logger.debug(f"Starting chat stream with message: {message}")
        
        try:
            async for response in self.conversation_agent.process_message(message, system_prompt, context):
                yield self._format_chunk(response)

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error in process_chat_stream: {str(e)}", exc_info=True)
            yield self._format_chunk({"type": "error", "content": str(e)})

    @staticmethod
    def _format_chunk(data: Dict[str, Any]) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"