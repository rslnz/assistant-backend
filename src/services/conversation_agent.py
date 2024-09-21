import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain.schema import SystemMessage
from langchain.tools import BaseTool

from src.models.chat_models import ConversationContext, Role
from src.models.prompt_structures import LLMProcessingState, Status, StatusEnum, ToolUse
from src.services.llm_service import LLMService
from src.services.message_preparer import MessagePreparer
from src.services.tag_processor import TagProcessor
from src.tools.datetime import DateTimeTool
from src.tools.math import MathTool
from src.tools.web_parse import WebParseTool
from src.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

class ConversationAgent:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.tools: List[BaseTool] = [
            WebSearchTool(),
            WebParseTool(),
            MathTool(),
            DateTimeTool(),
        ]
        self.message_preparer = MessagePreparer(self.tools)

    async def process_message(self, message: str, system_prompt: str, context: ConversationContext) -> AsyncGenerator[Dict[str, Any], None]:
        context.add_message(Role.HUMAN, message)

        state = LLMProcessingState()

        state.add_message(SystemMessage(content=system_prompt))
        state.messages.extend(self.message_preparer.prepare_messages(context, state, message))
        
        tag_processor = TagProcessor(
            stream_tags=["text"],
            buffer_tags=["tool", "summary", "plan", "reasoning", "status", "full_text"]
        )

        logger.debug(f"Starting to process message: {message}")
        async for event in self._process_conversation(tag_processor, state, context):
            yield event

        logger.debug("Finished processing message.")

    async def _process_conversation(self, tag_processor: TagProcessor, state: LLMProcessingState, context: ConversationContext) -> AsyncGenerator[Dict[str, Any], None]:
        extra_iterations = 3
        max_iterations = extra_iterations
        process_completed = False
        initial_message_count = len(state.messages)

        for iteration in range(25):
            if iteration >= max_iterations:
                yield self._format_response("error", "Response generation exceeded the maximum number of iterations.")
                break

            messages = self.message_preparer.prepare_messages(context, state, state.messages[-1].content if state.messages else "")
            llm_stream = self.llm_service.stream(messages)

            async for event in self._process_llm_response(tag_processor, llm_stream, state):
                yield event

            if not state.status:
                logger.warning("No STATUS set after processing LLM response. Setting default STATUS.")
                state.status = Status(status=StatusEnum.CONTINUE, reason="Continuing due to missing status")
                yield self._format_response("status", state.status.model_dump())

            if state.current_plan:
                max_iterations = state.current_plan.total_steps + extra_iterations
                logger.debug(f"Updated max_iterations to {max_iterations} based on current plan")

            if state.status and state.status.status == StatusEnum.COMPLETE and state.tool_queue:
                logger.warning("StatusEnum.COMPLETE was set, but tool_queue is not empty. Ignoring remaining tools.")
                state.tool_queue.clear()
                process_completed = True
                break

            if state.tool_queue:
                async for event in self._handle_tool_queue(state.tool_queue, state):
                    yield event
            
            if not state.status:
                logger.warning("No [STATUS] tag received from LLM. Assuming COMPLETE.")
                process_completed = True
                break

            if state.status.status in [StatusEnum.CLARIFY, StatusEnum.COMPLETE]:
                process_completed = True
                break

            if state.status.status == StatusEnum.CONTINUE:
                state.prepare_continuation_message()

            state.status = None

        if not process_completed:
            yield self._format_response("error", "Response generation did not complete within the maximum number of iterations.")

        self._update_context(context, state, initial_message_count)

        yield self._format_response("updated_context", context.model_dump())

    async def _process_llm_response(self, tag_processor: TagProcessor, llm_stream: AsyncGenerator[str, None], state: LLMProcessingState) -> AsyncGenerator[Dict[str, Any], None]:
        async for event in tag_processor.process_stream(llm_stream):
            if event['tag'] == 'plan':
                state.update_plan(event)
            elif event['tag'] == 'reasoning':
                user_notification = state.process_reasoning(event)
                yield self._format_response("reasoning", user_notification)
            elif event['tag'] == 'text':
                yield self._format_response("text", event['content'])
            elif event['tag'] == 'full_text':
                state.process_full_text(event)
            elif event['tag'] == 'tool':
                state.process_tool(event)
            elif event['tag'] == 'summary':
                state.process_summary(event)
            elif event['tag'] == 'status':
                state.process_status(event)
            elif event['tag'] == 'debug':
                logger.debug(f"Debug content: {event['content']}")
            else:
                error_message = f"Unexpected tag received: {event['tag']}"
                logger.error(error_message)
                yield self._format_response("error", error_message)
                return

    async def _handle_tool_queue(self, tool_queue: List[ToolUse], state: LLMProcessingState) -> AsyncGenerator[Dict[str, Any], None]:
        async for event in self._process_tool_queue(tool_queue):
            if event['type'] in ['tool_start', 'tool_end', 'error']:
                yield event

            if event['type'] == 'tool_end':
                result = {
                    "id": event['content']['id'],
                    "name": event['content']['name'],
                    "result": event['content']['result']
                }
                state.add_tool_result(result)
            elif event['type'] == 'error':
                error_result = {
                    "id": tool_queue[0].id,
                    "name": tool_queue[0].name,
                    "error": event['content']
                }
                state.add_tool_result(error_result)

        state.tool_queue.clear()

    def _update_context(self, context: ConversationContext, state: LLMProcessingState, initial_message_count: int):
        for msg in state.messages[initial_message_count:]:
            context.add_message(msg.type, msg.content)
        if state.latest_summary:
            context.set_summary(state.latest_summary)

    async def _process_tool_queue(self, tool_queue: List[ToolUse]) -> AsyncGenerator[Dict[str, Any], None]:
        for tool_use in tool_queue:
            tool = self._get_tool(tool_use.name)
            if not tool:
                error_message = f"Tool '{tool_use.name}' is not available. Please use a different tool."
                yield self._format_response("error", error_message)
                continue
            
            yield self._format_response("tool_start", {
                "id": tool_use.id,
                "name": tool.name,
                "description": tool.description,
                "user_notification": tool_use.user_notification
            })

            try:
                result = await self._execute_tool(tool, tool_use)
                yield self._format_response("tool_end", {
                    "id": tool_use.id,
                    "name": tool.name,
                    "result": result
                })
            except Exception as e:
                error_message = self.message_preparer.get_tool_error_message(tool_use.name, str(e))
                yield self._format_response("tool_end", {
                    "id": tool_use.id,
                    "name": tool.name,
                    "error": error_message
                })

    def _get_tool(self, tool_name: str) -> Optional[BaseTool]:
        return next((tool for tool in self.tools if tool.name == tool_name), None)

    async def _execute_tool(self, tool: BaseTool, tool_use: ToolUse) -> str:
        logger.debug(f"Executing tool: {tool.name}")
        logger.debug(f"Tool data: {tool_use.model_dump_json()}")

        arguments = tool_use.arguments
        logger.debug(f"Arguments being passed to tool: {json.dumps(arguments)}")
        
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
