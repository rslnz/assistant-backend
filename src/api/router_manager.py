from fastapi import APIRouter
from src.api.endpoints.root import RootEndpoints
from src.api.endpoints.openai import OpenAIEndpoints
from src.services.openai_service import OpenAIService

class APIRouterManager:
    def __init__(self, openai_service: OpenAIService):
        self.main_router = APIRouter()
        self.openai_service = openai_service
        self._init_routers()

    def _init_routers(self):
        root_endpoints = RootEndpoints()

        self.main_router.add_api_route("/", root_endpoints.root, methods=["GET"], tags=["root"])
        self.main_router.add_api_route("/health", root_endpoints.health_check, methods=["GET"], tags=["root"])

        openai_endpoints = OpenAIEndpoints(self.openai_service)

        self.main_router.add_api_route("/openai/chat", openai_endpoints.openai_chat_stream, methods=["POST"], tags=["openai"])

    def get_router(self) -> APIRouter:
        return self.main_router