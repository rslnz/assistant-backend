from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from langchain.schema import BaseMessage, SystemMessage, AIMessage
from enum import Enum
import uuid
import logging
import json

logger = logging.getLogger(__name__)

class Tool(BaseModel):
    name: str
    depends_on: List[str] = []

class Step(BaseModel):
    description: str
    status: str
    tools: List[Tool] = []

class Plan(BaseModel):
    steps: List[Step]
    current_step: int
    total_steps: int

class Reasoning(BaseModel):
    thought: str
    user_notification: str

class ToolUse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    arguments: dict
    user_notification: str

class StatusEnum(str, Enum):
    CONTINUE = "continue"
    CLARIFY = "clarify"
    COMPLETE = "complete"

class Status(BaseModel):
    status: str
    reason: Optional[str] = None

class PromptStructures(BaseModel):
    plan: Optional[Plan] = None
    reasoning: Optional[Reasoning] = None
    tool: Optional[ToolUse] = None
    text: Optional[str] = None
    summary: Optional[str] = None
    status: Optional[Status] = None

class LLMProcessingState(BaseModel):
    current_plan: Optional[Plan] = None
    reasoning_history: List[str] = Field(default_factory=list)
    tool_queue: List[ToolUse] = Field(default_factory=list)
    messages: List[BaseMessage] = Field(default_factory=list)
    summary: Optional[str] = None
    summary_history: List[str] = Field(default_factory=list)
    latest_summary: Optional[str] = None  # Новое поле для последней суммаризации
    status: Optional[Status] = None
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)

    def update_plan(self, event: Dict[str, Any]) -> None:
        try:
            new_plan = Plan.model_validate_json(event['content'])
            if self.current_plan is None or new_plan != self.current_plan:
                self.current_plan = new_plan
                logger.debug(f"Updated plan: {new_plan.model_dump_json()}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            logger.debug(f"Problematic content: {event['content']}")

    def process_reasoning(self, event: Dict[str, Any]) -> str:
        logger.debug(f"Processing reasoning: {event['content']}")
        try:
            reasoning = Reasoning.model_validate_json(event['content'])
            self.reasoning_history.append(reasoning.thought)
            logger.debug(f"Reasoning: {reasoning.model_dump_json()}")
            return reasoning.user_notification
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse reasoning JSON: {e}")
            logger.debug(f"Problematic content: {event['content']}")
            return ""

    def process_tool(self, event: Dict[str, Any]) -> None:
        try:
            tool_use = ToolUse.model_validate_json(event['content'])
            self.tool_queue.append(tool_use)
            logger.debug(f"Tool added to queue: {tool_use.model_dump_json()}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool JSON: {e}")
            logger.debug(f"Problematic content: {event['content']}")

    def process_full_text(self, event: Dict[str, Any]) -> None:
        self.messages.append(AIMessage(content=event['content']))
        logger.debug(f"Full text received: {event['content']}")

    def process_summary(self, event: Dict[str, Any]) -> None:
        self.summary_history.append(event['content'])
        self.latest_summary = event['content']
        self.summary = f"Previous summaries: {' | '.join(self.summary_history[:-1])}\nLatest summary: {self.latest_summary}"
        logger.debug(f"Summary received: {self.summary}")
        logger.debug(f"Latest summary: {self.latest_summary}")

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        logger.debug(f"Message added: {message.model_dump_json()}")

    def add_tool_result(self, result: Dict[str, Any]) -> None:
        self.tool_results.append(result)
        logger.debug(f"Tool result added: {result}")

    def prepare_continuation_message(self) -> None:
        tool_results_summary = []
        for result in self.tool_results:
            if 'error' in result:
                tool_results_summary.append(f"Tool {result['name']} (ID: {result['id']}) error: {result['error']}")
            else:
                tool_results_summary.append(f"Tool {result['name']} (ID: {result['id']}) result: {result['result']}")

        continuation_message = (
            f"Continue with the next step of your plan, summarizing previous reasoning steps and tool results. "
            f"Current plan: {self.current_plan.model_dump_json() if self.current_plan else 'No plan'}\n"
            f"Previous reasoning: {' '.join(self.reasoning_history)}\n"
            f"Tool results: {' | '.join(tool_results_summary)}"
        )

        self.messages.append(SystemMessage(content=continuation_message))
        logger.debug(f"Continuation message added: {continuation_message}")

        self.tool_results.clear()

    def process_status(self, event: Dict[str, Any]) -> None:
        try:
            self.status = Status.model_validate_json(event['content'])
            logger.info(f"Status processed: {self.status.model_dump_json()}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse status JSON: {e}")
            logger.debug(f"Problematic content: {event['content']}")
        except ValidationError as e:
            logger.error(f"Failed to validate Status: {e}")
            logger.debug(f"Problematic content: {event['content']}")