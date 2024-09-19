import logging
from fastapi.responses import StreamingResponse
from src.models.chat_models import ChatRequest
from src.services.openai_service import OpenAIService
from src.api.utils import handle_exceptions

logger = logging.getLogger(__name__)

class OpenAIEndpoints:
    def __init__(self, openai_service: OpenAIService):
        self.openai_service = openai_service

    @handle_exceptions
    async def openai_chat_stream(self, request: ChatRequest) -> StreamingResponse:
        """
        Processes a chat request and returns a streaming response.

        Args:
            request (ChatRequest): Chat request containing the user's message and conversation context.

        Returns:
            StreamingResponse: A streaming response where each chunk has the format:
                data: {
                    "type": str,    # Chunk type (e.g., "text", "tool_start", "tool_end", "context_update", "error")
                    "content": str  # Chunk content
                }

        The stream ends with the message:
        data: [DONE]

        Example chunks in the stream:
        data: {"type": "text", "content": "Hello"}
        data: {"type": "tool_start", "content": {"name": "web_search", "description": "Search for information on the Internet", "action": "Searching for weather information in your city"}}
        data: {"type": "tool_end", "content": {"name": "web_search", "result": "Weather in your city: 20 degrees, sunny, no precipitation"}}
        data: {"type": "context_update", "content": {...}}  # Full updated context
        data: {"type": "error", "content": "Error processing request"}

        Note:
        - Each chunk represents a separate event in the request processing.
        - The client should handle different types of chunks accordingly.
        - LLM tokens are transmitted in real-time to display the response as it's being generated.
        - The context is updated at the end of the stream, reflecting the final state of the conversation.
        """
        async def stream_generator():
            async for chunk in self.openai_service.process_chat_stream(request):
                yield chunk
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
