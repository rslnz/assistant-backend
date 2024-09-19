import json
from typing import Dict, Any, List, AsyncGenerator, Optional
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import BaseTool
from src.tools import MathTool, WebSearchTool, WebParseTool
from src.models.chat_models import ConversationContext, Role
from src.services.llm_service import LLMService
from src.services.tag_processor import TagProcessor
from src.services.message_preparer import MessagePreparer

import logging
logger = logging.getLogger(__name__)

class ConversationAgent:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.tools: List[BaseTool] = [
            MathTool(),
            WebSearchTool(),
            WebParseTool()
        ]
        self.message_preparer = MessagePreparer(self.tools)

    async def process_message(self, message: str, system_prompt: str, context: ConversationContext) -> AsyncGenerator[Dict[str, Any], None]:
        context.add_message(Role.HUMAN, message)
        messages = self.message_preparer.prepare_messages(context, system_prompt, message)
        
        tag_processor = TagProcessor(
            stream_tags=["content"],
            buffer_tags=["tool", "summary"]
        )

        logger.debug(f"Starting to process message: {message}")
        async for event in self._process_conversation(tag_processor, messages, context):
            logger.debug(f"Generated event: {event}")
            yield event

        logger.debug("Finished processing message.")

    async def _process_conversation(self, tag_processor: TagProcessor, messages: List[BaseMessage], context: ConversationContext) -> AsyncGenerator[Dict[str, Any], None]:
        while True:
            response_complete = True
            
            async for event in tag_processor.process_stream(self.llm_service.stream(messages)):
                logger.debug(f"Processing event: {event}")
                if event['tag'] == 'content':
                    yield self._format_response("text", event['content'])
                elif event['tag'] == 'tool':
                    response_complete = False
                    async for tool_event in self._process_tool(event['content'], context):
                        yield tool_event
                    messages = self.message_preparer.prepare_messages(context, messages[0].content, "Continue the conversation based on the tool results.")
                elif event['tag'] == 'summary':
                    context.set_summary(event['content'])
                elif event['tag'] == 'content_full':
                    context.add_message(Role.AI, event['content'])
                elif event['tag'] == 'debug':
                    logger.debug(f"Full debug content: {event['content']}")
                elif event['tag'] == 'error':
                    context.add_message(Role.SYSTEM, event['content'])
                    yield self._format_response("error", event['content'])

            if response_complete:
                break
        
        yield self._format_response("context_update", context.model_dump())

    async def _process_tool(self, tool_content: Dict[str, Any], context: ConversationContext) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            tool_name = tool_content.get("name")
            tool = self._get_tool(tool_name)
            if not tool:
                yield self._format_response("error", f"Tool '{tool_name}' is not available.")
                return

            yield self._format_response("tool_start", {
                "name": tool.name,
                "description": tool.description,
                "action": tool_content.get("action", f"Using tool '{tool.name}'...")
            })
            
            result = await self._execute_tool(tool, tool_content)
            
            yield self._format_response("tool_end", {
                "name": tool.name,
                "result": result
            })

            context.add_message(Role.SYSTEM, f"Tool {tool.name} result: {result}")

        except Exception as e:
            logger.error(f"Error processing tool: {str(e)}")
            yield self._format_response("error", f"Error processing tool: {str(e)}")

    def _get_tool(self, tool_name: str) -> Optional[BaseTool]:
        return next((tool for tool in self.tools if tool.name == tool_name), None)

    async def _execute_tool(self, tool: BaseTool, tool_data: Dict[str, Any]) -> str:
        try:
            arguments = tool_data.get("arguments", {})
            tool_input = json.dumps(arguments)
            return await tool.arun(tool_input)
        except Exception as e:
            error_message = f"Error using tool {tool.name}: {str(e)}"
            logger.error(error_message)
            return error_message

    def _format_response(self, response_type: str, content: Any) -> Dict[str, Any]:
        return {
            "type": response_type,
            "content": content
        }
