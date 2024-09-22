import traceback
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain.tools import BaseTool
from pydantic import ValidationError

from src.models.conversation_state import ConversationState
from src.models.prompt_structures import Plan, Reasoning, Status, StatusEnum, ToolUse
from src.models.tags import Tag
from src.services.llm_service import LLMService
from src.services.message_preparer import MessagePreparer
from src.services.tag_processor import TagConfig, TagProcessingMode, TagProcessor
from src.tools.math import MathTool
from src.tools.web_parse import WebParseTool
from src.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

class ConversationError(Exception):
    pass

class ConversationManager:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.tools = [WebSearchTool(), WebParseTool(), MathTool()]
        self.tool_manager = ToolManager(self.tools)
        self.message_preparer = MessagePreparer(self.tools)
        self.formatter = ResponseFormatter()
        self.iteration_controller = IterationController()

    async def process_message(self, message: str, system_prompt: str, context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        logger.debug(f"Starting to process message: '{message}'")

        try:
            state = ConversationState.from_context(context, system_prompt, message)
            async for event in self._process_conversation(state):
                logger.debug(f"Event: {event}")
                yield event
        except ConversationError as e:
            logger.error(f"Conversation error: {str(e)}")
            yield self.formatter.format_response(Tag.ERROR, str(e))
            print(traceback.format_exc())
        except Exception as e:
            logger.error(f"Unexpected error during conversation processing: {str(e)}")
            yield self.formatter.format_response(Tag.ERROR, f"An unexpected error occurred: {str(e)}")
            print(traceback.format_exc())

        logger.debug("Finished processing message.")

    async def _process_conversation(self, state: ConversationState) -> AsyncGenerator[Dict[str, Any], None]:
        async for event in self._conversation_loop(state):
            yield event

        if self.iteration_controller.is_conversation_complete(state):
            yield self.formatter.format_response(Tag.UPDATED_CONTEXT, state.get_updated_context().model_dump())
        else:
            raise ConversationError("Conversation did not complete properly.")

    async def _conversation_loop(self, state: ConversationState) -> AsyncGenerator[Dict[str, Any], None]:
        iteration = 0
        while not self.iteration_controller.should_terminate(iteration, state):
            iteration += 1

            messages = self.message_preparer.prepare_messages(state)
            logger.debug(f"Prepared messages: {messages}")

            llm_stream = self.llm_service.stream(messages)

            async for event in self._process_llm_response(llm_stream, state):
                yield event

            if self._handle_missing_status(state):
                raise ConversationError("No STATUS set after processing LLM response.")

            if self.iteration_controller.is_conversation_complete(state):
                return

            self.iteration_controller.update_max_iterations(state)

            state.prepare_for_next_iteration()

        if not self.iteration_controller.is_conversation_complete(state):
            raise ConversationError(f"Response generation did not complete within the maximum number of iterations ({state.max_iterations}).")

    def _handle_missing_status(self, state: ConversationState) -> bool:
        return not state.status

    async def _process_llm_response(self, llm_stream: AsyncGenerator[str, None], state: ConversationState) -> AsyncGenerator[Dict[str, Any], None]:
        tag_processor = TagProcessor(self._create_tag_config())

        async for event in tag_processor.process_stream(llm_stream):
            tag = Tag(event['tag'])
            content = event['content']

            if tag == Tag.PLAN:
                try:
                    new_plan = Plan.model_validate_json(content)
                    if state.current_plan != new_plan:
                        state.update_plan(new_plan)
                        logger.debug(f"Updated plan: {new_plan.model_dump_json()}")
                except ValidationError as e:
                    raise ConversationError(f"Failed to validate Plan: {e}")

            elif tag == Tag.REASONING:
                try:
                    reasoning = Reasoning.model_validate_json(content)
                    state.add_reasoning(reasoning.thought)
                    logger.debug(f"Reasoning: {reasoning.model_dump_json()}")
                    yield self.formatter.format_response(Tag.REASONING, reasoning.user_notification)
                except ValidationError as e:
                    raise ConversationError(f"Failed to validate Reasoning: {e}")

            elif tag == Tag.TOOL:
                try:
                    logger.debug(f"Tool call received: {content}")
                    tool_use = ToolUse.model_validate_json(content)
                    tool = self.tool_manager.get_tool(tool_use.name)
                    if tool is None:
                        raise ConversationError(f"Tool not found: {tool_use.name}")

                    yield self.formatter.format_response(Tag.TOOL_START, {
                        "id": tool_use.id,
                        "name": tool_use.name,
                        "description": tool.description,
                        "user_notification": tool_use.user_notification
                    })

                    result = await self.tool_manager.execute_tool(tool, tool_use)

                    tool_result = {
                        "id": tool_use.id,
                        "name": tool_use.name,
                        "result": result
                    }
                    state.add_tool_result(tool_result)
                    yield self.formatter.format_response(Tag.TOOL_END, tool_result)

                    logger.debug(f"Tool execution result: {tool_result}")
                except ValidationError as e:
                    raise ConversationError(f"Failed to validate ToolUse: {e}")
                except Exception as e:
                    error_message = f"Error executing tool: {str(e)}"
                    if 'tool_use' in locals():
                        error_result = {
                            "id": tool_use.id,
                            "name": tool_use.name,
                            "error": error_message
                        }
                        state.add_tool_result(error_result)
                        yield self.formatter.format_response(Tag.TOOL_END, error_result)
                        logger.debug(f"Tool execution error: {error_result}")
                    else:
                        raise ConversationError(error_message)

            elif tag == Tag.FULL_TEXT:
                state.add_ai_message(content)
                logger.debug(f"Full text received: {content}")

            elif tag == Tag.SUMMARY:
                state.update_summary(content)
                logger.debug(f"Summary received: {state.summary}")

            elif tag == Tag.STATUS:
                try:
                    status = Status.model_validate_json(content)
                    state.update_status(status)
                    logger.info(f"Status processed: {status.model_dump_json()}")
                except ValidationError as e:
                    raise ConversationError(f"Failed to validate Status: {e}")

            elif tag == Tag.DEBUG:
                logger.debug(f"Debug content: {content}")

            elif tag == Tag.TEXT:
                yield self.formatter.format_response(Tag.TEXT, content)

            else:
                raise ConversationError(f"Unexpected tag received: {tag}")

    def _create_tag_config(self) -> Dict[Tag, TagConfig]:
        return {
            Tag.PLAN: TagConfig(mode=TagProcessingMode.BUFFER),
            Tag.REASONING: TagConfig(mode=TagProcessingMode.BUFFER),
            Tag.TEXT: TagConfig(mode=TagProcessingMode.STREAM_AND_BUFFER, buffer_tag=Tag.FULL_TEXT),
            Tag.TOOL: TagConfig(mode=TagProcessingMode.BUFFER),
            Tag.STATUS: TagConfig(mode=TagProcessingMode.BUFFER),
            Tag.SUMMARY: TagConfig(mode=TagProcessingMode.BUFFER),
            Tag.DEBUG: TagConfig(mode=TagProcessingMode.BUFFER)
        }

class IterationController:
    def __init__(self, extra_iterations: int = 1):
        self.extra_iterations = extra_iterations

    def should_terminate(self, iteration: int, state: ConversationState) -> bool:
        return iteration >= state.max_iterations

    def update_max_iterations(self, state: ConversationState):
        if state.current_plan:
            state.max_iterations = state.current_plan.total_steps + self.extra_iterations
            logger.debug(f"Updated max_iterations to {state.max_iterations} based on current plan")

    def is_conversation_complete(self, state: ConversationState) -> bool:
        return state.status and state.status.status in [StatusEnum.CLARIFY, StatusEnum.COMPLETE]

class ResponseFormatter:
    @staticmethod
    def format_response(response_type: Tag, content: Any) -> Dict[str, Any]:
        return {
            "type": response_type,
            "content": content
        }

class ToolManager:
    def __init__(self, tools: List[BaseTool]):
        self.tools = tools

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        return next((tool for tool in self.tools if tool.name == tool_name), None)

    async def execute_tool(self, tool: BaseTool, tool_use: ToolUse) -> str:
        logger.debug(f"Executing tool: {tool.name}. Data: {tool_use.model_dump_json()}")
        try:
            return await tool.arun(tool_use.arguments)
        except Exception as e:
            raise ConversationError(f"Error executing tool {tool.name}: {str(e)}")
