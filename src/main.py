from fastapi import FastAPI

from src.api.router_manager import APIRouterManager
from src.config import settings
from src.services.conversation_agent import ConversationManager
from src.services.llm_service import LLMService
from src.services.openai_service import OpenAIService

app = FastAPI(
    title="Chat API",
    description="API for chat using OpenAI",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

llm_service = LLMService()
conversation_manager = ConversationManager(llm_service)
openai_service = OpenAIService(conversation_manager)

router_manager = APIRouterManager(openai_service)
app.include_router(router_manager.get_router())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)