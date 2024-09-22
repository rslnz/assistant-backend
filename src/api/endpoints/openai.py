import logging

from fastapi.responses import StreamingResponse

from src.api.utils import handle_exceptions
from src.models.chat_models import ChatRequest
from src.services.openai_service import OpenAIService

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
                    "type": str,    # Chunk type (e.g., "reasoning", "text", "tool_start", "tool_end", "context_update", "error")
                    "content": Any  # Chunk content, structure depends on the type
                }

        The stream ends with the message:
        data: [DONE]

        Example chunks in the stream:
        data: {"type": "reasoning", "content": "Searching"}
        data: {"type": "text", "content": "Here's what I found:"}
        data: {"type": "tool_start", "content": {"id": "unique_tool_call_id", "name": "web_search", "description": "Search for information on the Internet", "user_notification": "Searching for weather information"}}
        data: {"type": "tool_end", "content": {"id": "unique_tool_call_id", "name": "web_search", "result": "Weather: 20 degrees, sunny"}}
        data: {"type": "context_update", "content": {...}}
        data: {"type": "error", "content": "Error processing request"}

        Note:
        - Each chunk represents a separate event in the request processing.
        - The client should handle different types of chunks accordingly.
        - LLM tokens are transmitted in real-time to display the response as it's being generated.
        - The context_update event contains the full context of the entire conversation.
          This updated context must be sent with the next request to maintain conversation continuity.
        - The 'id' in tool_start and tool_end events represents a unique identifier for each specific tool call,
          allowing clients to match the start and end of individual tool executions.
        """
        async def stream_generator():
            async for chunk in self.openai_service.process_chat_stream(request):
                yield chunk
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
