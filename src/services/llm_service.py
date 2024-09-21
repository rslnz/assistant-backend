import logging
from typing import AsyncGenerator, List

from langchain.schema import BaseMessage
from langchain_openai import ChatOpenAI

from src.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=settings.OPENAI_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
            streaming=True,
        )

    async def generate(self, messages: List[BaseMessage]) -> str:
        response = await self.llm.agenerate([messages])
        return response.generations[0][0].text.strip()

    async def stream(self, messages: List[BaseMessage]) -> AsyncGenerator[str, None]:
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content