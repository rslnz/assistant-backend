import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain.tools import BaseTool
from pydantic import ValidationError

from src.models.chat_models import ConversationContext, Role
from src.models.prompt_structures import (
    LLMProcessingState,
    Plan,
    Reasoning,
    Status,
    StatusEnum,
    ToolUse,
)
from src.models.tags import Tag
from src.services.llm_service import LLMService
from src.services.message_preparer import MessagePreparer
from src.services.tag_processor import TagConfig, TagProcessingMode, TagProcessor
from src.tools.datetime import DateTimeTool
from src.tools.math import MathTool
from src.tools.web_parse import WebParseTool
from src.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.tools = [WebSearchTool(), WebParseTool(), MathTool(), DateTimeTool()]
        self.tool_manager = ToolManager(self.tools)
        self.message_preparer = MessagePreparer(self.tools)
        self.formatter = ResponseFormatter()
        self.tag_handler = TagHandler(self.formatter, self.message_preparer)
        self.context_updater = ContextUpdater()
        self.iteration_controller = IterationController()
        self.max_tool_retries = 2

    async def process_message(self, message: str, system_prompt: str, context: ConversationContext) -> AsyncGenerator[Dict[str, Any], None]:
        context.add_message(Role.HUMAN, message)
        state = self._initialize_state(system_prompt, context, message)
        tag_processor = self._create_tag_processor()

        logger.debug(f"Starting to process message: {message}")
        async for event in self._process_conversation(tag_processor, state, context):
            yield event
        logger.debug("Finished processing message.")

    def _initialize_state(self, system_prompt: str, context: ConversationContext, message: str) -> LLMProcessingState:
        state = LLMProcessingState()
        state.add_system_message(system_prompt)
        state.messages.extend(self.message_preparer.prepare_messages(context, state, message))
        return state

    def _create_tag_processor(self) -> TagProcessor:
        tag_config = {
            Tag.PLAN: TagConfig(mode=TagProcessingMode.BUFFER),
            Tag.REASONING: TagConfig(mode=TagProcessingMode.BUFFER),
            Tag.TEXT: TagConfig(mode=TagProcessingMode.STREAM_AND_BUFFER, buffer_tag=Tag.FULL_TEXT),
            Tag.TOOL: TagConfig(mode=TagProcessingMode.BUFFER),
            Tag.STATUS: TagConfig(mode=TagProcessingMode.BUFFER),
            Tag.SUMMARY: TagConfig(mode=TagProcessingMode.BUFFER),
            Tag.DEBUG: TagConfig(mode=TagProcessingMode.BUFFER)
        }
        return TagProcessor(tag_config)

    async def _process_conversation(self, tag_processor: TagProcessor, state: LLMProcessingState, context: ConversationContext) -> AsyncGenerator[Dict[str, Any], None]:
        initial_message_count = len(state.messages)

        async for event in self._conversation_loop(tag_processor, state, context):
            yield event

        if self.iteration_controller.is_conversation_complete(state):
            self.context_updater.update_context(context, state, initial_message_count)
            yield self.formatter.format_response(Tag.UPDATED_CONTEXT, context.model_dump())
        else:
            yield self.formatter.format_response(Tag.ERROR, "Conversation did not complete properly.")

    async def _conversation_loop(self, tag_processor: TagProcessor, state: LLMProcessingState, context: ConversationContext) -> AsyncGenerator[Dict[str, Any], None]:
        iteration = 0
        while not self.iteration_controller.should_terminate(iteration, state):
            iteration += 1

            messages = self.message_preparer.prepare_messages(context, state, state.messages[-1].content if state.messages else "")
            llm_stream = self.llm_service.stream(messages)

            has_error = False
            async for event in self._process_llm_response(tag_processor, llm_stream, state):
                yield event
                if event['type'] == Tag.ERROR:
                    has_error = True
            if has_error:
                return

            missing_status_error = self._handle_missing_status(state)
            if missing_status_error:
                yield missing_status_error
                return

            if self.iteration_controller.is_conversation_complete(state):
                return

            self.iteration_controller.update_max_iterations(state)

            if state.tool_queue:
                async for event in self._handle_tool_queue(state.tool_queue, state):
                    yield event
                    if event['type'] == Tag.ERROR:
                        return

            self.iteration_controller.prepare_for_next_iteration(state)

        if not self.iteration_controller.is_conversation_complete(state):
            yield self.formatter.format_response(Tag.ERROR, f"Response generation did not complete within the maximum number of iterations ({state.max_iterations}).")

    def _handle_missing_status(self, state: LLMProcessingState) -> Optional[Dict[str, Any]]:
        if not state.status:
            error_message = "No STATUS set after processing LLM response. This is a critical error."
            logger.error(error_message)
            return self.formatter.format_response(Tag.ERROR, error_message)
        return None

    async def _process_llm_response(self, tag_processor: TagProcessor, llm_stream: AsyncGenerator[str, None], state: LLMProcessingState) -> AsyncGenerator[Dict[str, Any], None]:
        tag_handlers = {
            Tag.PLAN: self.tag_handler.handle_plan_tag,
            Tag.REASONING: self.tag_handler.handle_reasoning_tag,
            Tag.TOOL: self.tag_handler.handle_tool_tag,
            Tag.FULL_TEXT: self.tag_handler.handle_full_text_tag,
            Tag.SUMMARY: self.tag_handler.handle_summary_tag,
            Tag.STATUS: self.tag_handler.handle_status_tag,
            Tag.DEBUG: self.tag_handler.handle_debug_tag,
            Tag.TEXT: self.tag_handler.handle_text_tag
        }

        async for event in tag_processor.process_stream(llm_stream):
            try:
                tag = Tag(event['tag'])
            except ValueError:
                error_message = f"Invalid tag: {event['tag']}"
                logger.error(error_message)
                yield self.formatter.format_response(Tag.ERROR, error_message)
                continue

            handler = tag_handlers.get(tag, self.tag_handler.handle_unknown_tag)
            result = await handler(event, state)
            if result is not None:
                yield result

    async def _handle_tool_queue(self, tool_queue: List[ToolUse], state: LLMProcessingState) -> AsyncGenerator[Dict[str, Any], None]:
        for tool_use in tool_queue:
            async for event in self._process_single_tool(tool_use):
                yield event

                if event['type'] == Tag.TOOL_END:
                    state.add_tool_result(event['content'])
                elif event['type'] == Tag.ERROR:
                    state.add_tool_result({
                        "id": tool_use.id,
                        "name": tool_use.name,
                        "error": event['content']
                    })
        tool_queue.clear()

    async def _process_single_tool(self, tool_use: ToolUse) -> AsyncGenerator[Dict[str, Any], None]:
        tool = self.tool_manager.get_tool(tool_use.name)
        if not tool:
            yield self.formatter.format_response(Tag.ERROR, f"Tool '{tool_use.name}' is not available. Please use a different tool.")
            return

        yield self.formatter.format_response(Tag.TOOL_START, {
            "id": tool_use.id,
            "name": tool.name,
            "description": tool.description,
            "user_notification": tool_use.user_notification
        })

        try:
            result = await self.tool_manager.execute_tool(tool, tool_use)
            yield self.formatter.format_response(Tag.TOOL_END, {
                "id": tool_use.id,
                "name": tool.name,
                "result": result
            })
        except Exception as e:
            error_message = self.message_preparer.get_tool_error_message(tool_use.name, str(e))
            yield self.formatter.format_response(Tag.TOOL_END, {
                "id": tool_use.id,
                "name": tool.name,
                "error": error_message
            })

class ContextUpdater:
    def update_context(self, context: ConversationContext, state: LLMProcessingState, initial_message_count: int):
        for msg in state.messages[initial_message_count:]:
            if msg.type == "human" or msg.type == "ai":
                context.add_message(msg.type, msg.content)

        latest_summary = self.create_latest_summary(state)
        context.set_summary(latest_summary)
        
        logger.info("Context updated with new messages and latest summary.")

    def create_latest_summary(self, state: LLMProcessingState) -> str:
        summary_parts = []
        
        if state.latest_summary:
            summary_parts.append(state.latest_summary)
        
        if state.current_plan:
            current_step = state.current_plan.steps[state.current_plan.current_step - 1]
            summary_parts.append(f"Current task: {current_step.description}")
        
        if state.status:
            summary_parts.append(f"Status: {state.status.status.value}")
        
        return " | ".join(summary_parts)

class IterationController:
    def __init__(self, extra_iterations: int = 1):
        self.extra_iterations = extra_iterations

    def should_terminate(self, iteration: int, state: LLMProcessingState) -> bool:
        return iteration >= state.max_iterations

    def update_max_iterations(self, state: LLMProcessingState):
        if state.current_plan:
            state.max_iterations = state.current_plan.total_steps + self.extra_iterations
            logger.debug(f"Updated max_iterations to {state.max_iterations} based on current plan")

    def is_conversation_complete(self, state: LLMProcessingState) -> bool:
        return state.status and state.status.status in [StatusEnum.CLARIFY, StatusEnum.COMPLETE]
    
    def prepare_for_next_iteration(self, state: LLMProcessingState):
        if state.status and state.status.status == StatusEnum.CONTINUE:
            state.prepare_continuation_message()
        state.status = None

class ResponseFormatter:
    @staticmethod
    def format_response(response_type: Tag, content: Any) -> Dict[str, Any]:
        return {
            "type": response_type,
            "content": content
        }

class TagHandler:
    def __init__(self, formatter: ResponseFormatter, message_preparer: MessagePreparer):
        self.formatter = formatter
        self.message_preparer = message_preparer

    async def handle_plan_tag(self, event: Dict[str, Any], state: LLMProcessingState) -> Optional[Dict[str, Any]]:
        try:
            new_plan = Plan.model_validate_json(event['content'])
            if state.current_plan is None or new_plan != state.current_plan:
                state.current_plan = new_plan
                logger.debug(f"Updated plan: {new_plan.model_dump_json()}")
        except ValidationError as e:
            error_message = f"Failed to validate Plan: {e}"
            logger.error(error_message)
            logger.debug(f"Problematic content: {event['content']}")
            return self.formatter.format_response(Tag.ERROR, error_message)

    async def handle_reasoning_tag(self, event: Dict[str, Any], state: LLMProcessingState) -> Dict[str, Any]:
        logger.debug(f"Processing reasoning: {event['content']}")
        try:
            reasoning = Reasoning.model_validate_json(event['content'])
            state.reasoning_history.append(reasoning.thought)
            logger.debug(f"Reasoning: {reasoning.model_dump_json()}")
            return self.formatter.format_response(Tag.REASONING, reasoning.user_notification)
        except ValidationError as e:
            error_message = f"Failed to validate Reasoning: {e}"
            logger.error(error_message)
            logger.debug(f"Problematic content: {event['content']}")
            return self.formatter.format_response(Tag.ERROR, error_message)

    async def handle_tool_tag(self, event: Dict[str, Any], state: LLMProcessingState) -> Optional[Dict[str, Any]]:
        try:
            tool_use = ToolUse.model_validate_json(event['content'])
            state.tool_queue.append(tool_use)
            logger.debug(f"Tool added to queue: {tool_use.model_dump_json()}")
        except ValidationError as e:
            error_message = f"Failed to validate ToolUse: {e}"
            logger.error(error_message)
            logger.debug(f"Problematic content: {event['content']}")
            return self.formatter.format_response(Tag.ERROR, error_message)

    async def handle_full_text_tag(self, event: Dict[str, Any], state: LLMProcessingState) -> None:
        state.add_ai_message(event['content'])
        logger.debug(f"Full text received: {event['content']}")

    async def handle_summary_tag(self, event: Dict[str, Any], state: LLMProcessingState) -> None:
        state.summary_history.append(event['content'])
        state.latest_summary = event['content']
        state.summary = f"Previous summaries: {' | '.join(state.summary_history[:-1])}\nLatest summary: {state.latest_summary}"
        logger.debug(f"Summary received: {state.summary}")
        logger.debug(f"Latest summary: {state.latest_summary}")

    async def handle_status_tag(self, event: Dict[str, Any], state: LLMProcessingState) -> Optional[Dict[str, Any]]:
        try:
            state.status = Status.model_validate_json(event['content'])
            logger.info(f"Status processed: {state.status.model_dump_json()}")
        except ValidationError as e:
            error_message = f"Failed to validate Status: {e}"
            logger.error(error_message)
            logger.debug(f"Problematic content: {event['content']}")
            return self.formatter.format_response(Tag.ERROR, error_message)

    async def handle_debug_tag(self, event: Dict[str, Any], _: LLMProcessingState) -> None:
        logger.debug(f"Debug content: {event['content']}")

    async def handle_unknown_tag(self, event: Dict[str, Any], _: LLMProcessingState) -> Dict[str, Any]:
        error_message = f"Unexpected tag received: {event['tag']}"
        logger.error(error_message)
        return self.formatter.format_response(Tag.ERROR, error_message)

    async def handle_text_tag(self, event: Dict[str, Any], _: LLMProcessingState) -> Dict[str, Any]:
        return self.formatter.format_response(Tag.TEXT, event['content'])

class ToolManager:
    def __init__(self, tools: List[BaseTool]):
        self.tools = tools

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        return next((tool for tool in self.tools if tool.name == tool_name), None)

    async def execute_tool(self, tool: BaseTool, tool_use: ToolUse) -> str:
        logger.debug(f"Executing tool: {tool.name}")
        logger.debug(f"Tool data: {tool_use.model_dump_json()}")

        arguments = tool_use.arguments
        logger.debug(f"Arguments being passed to tool: {json.dumps(arguments)}")
        
        try:
            return await tool.arun(arguments)
        except Exception as e:
            logger.error(f"Error executing tool {tool.name}: {str(e)}", exc_info=True)
            raise
