import json
from typing import Dict, Any, List, AsyncGenerator, Optional
from langchain.schema import BaseMessage
from langchain.tools import BaseTool
from src.tools import WebSearchTool, WebParseTool, MathTool
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
            WebSearchTool(),
            WebParseTool(),
            MathTool()
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
            if event['type'] not in ["text"]:
                logger.debug(f"Generated event: {event}")
            yield event

        logger.debug("Finished processing message.")

    async def _process_conversation(self, tag_processor: TagProcessor, messages: List[BaseMessage], context: ConversationContext) -> AsyncGenerator[Dict[str, Any], None]:
        max_iterations = 5
        for _ in range(max_iterations):
            response_complete = True
            try:
                async for event in tag_processor.process_stream(self.llm_service.stream(messages)):
                    if event['tag'] == 'content':
                        yield self._format_response("text", event['content'])
                    elif event['tag'] == 'tool':
                        response_complete = False
                        async for tool_event in self._process_tool(event['content'], context):
                            yield tool_event
                        messages = self.message_preparer.prepare_messages(context, messages[0].content, "Continue the conversation based on the tool results.")
                        break
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
            except Exception as e:
                logger.error(f"Error in conversation processing: {str(e)}")
                yield self._format_response("error", f"An error occurred: {str(e)}")
                break

        yield self._format_response("context_update", context.model_dump())

    async def _process_tool(self, tool_content: Dict[str, Any], context: ConversationContext) -> AsyncGenerator[Dict[str, Any], None]:
        tool_name = tool_content.get("name")
        logger.debug(f"Processing tool: {tool_name}")
        logger.debug(f"Tool content: {json.dumps(tool_content, indent=2)}")
        
        tool = self._get_tool(tool_name)
        if not tool:
            error_message = f"Tool '{tool_name}' is not available. Please use a different tool."
            yield self._format_response("error", error_message)
            context.add_message(Role.SYSTEM, error_message)
            return

        yield self._format_response("tool_start", {
            "name": tool.name,
            "description": tool.description,
            "action": tool_content.get("action", f"Using tool '{tool.name}'...")
        })
        
        try:
            result = await self._execute_tool(tool, tool_content)
            logger.debug(f"Tool result: {result}")
            
            yield self._format_response("tool_end", {
                "name": tool.name,
                "result": result
            })

            context.add_message(Role.SYSTEM, f"Tool {tool.name} result: {result}")
        except Exception as e:
            error_message = self.message_preparer.get_tool_error_message(tool_name, str(e))
            logger.error(f"Error in _process_tool: {error_message}")
            yield self._format_response("error", error_message)
            context.add_message(Role.SYSTEM, error_message)

    def _get_tool(self, tool_name: str) -> Optional[BaseTool]:
        return next((tool for tool in self.tools if tool.name == tool_name), None)

    async def _execute_tool(self, tool: BaseTool, tool_data: Dict[str, Any]) -> str:
        logger.debug(f"Executing tool: {tool.name}")
        logger.debug(f"Tool data: {json.dumps(tool_data, indent=2)}")

        arguments = tool_data.get("arguments", {})
        logger.debug(f"Original arguments: {json.dumps(arguments, indent=2)}")
        if not isinstance(arguments, dict):
            arguments = {"query": str(arguments)}
        
        logger.debug(f"Final arguments being passed to tool: {json.dumps(arguments, indent=2)}")
        
        try:
            return await tool.arun(arguments)
        except Exception as e:
            logger.error(f"Error executing tool {tool.name}: {str(e)}", exc_info=True)
            raise

    def _format_response(self, response_type: str, content: Any) -> Dict[str, Any]:
        return {
            "type": response_type,
            "content": content
        }
