import logging
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain.schema import AIMessage, BaseMessage, SystemMessage
from pydantic import BaseModel, Field

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
    arguments: Dict[str, Any]
    user_notification: str

class StatusEnum(str, Enum):
    CONTINUE = "continue"
    CLARIFY = "clarify"
    COMPLETE = "complete"

class Status(BaseModel):
    status: StatusEnum
    reason: Optional[str] = None

class LLMProcessingState(BaseModel):
    current_plan: Optional[Plan] = None
    reasoning_history: List[str] = Field(default_factory=list)
    tool_queue: List[ToolUse] = Field(default_factory=list)
    messages: List[BaseMessage] = Field(default_factory=list)
    summary: Optional[str] = None
    summary_history: List[str] = Field(default_factory=list)
    latest_summary: Optional[str] = None
    status: Optional[Status] = None
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    max_iterations: int = 3
    
    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        logger.debug(f"Message added: {message.content}")

    def add_ai_message(self, content: str) -> None:
        message = AIMessage(content=content)
        self.messages.append(message)
        logger.debug(f"AI message added: {content}")

    def add_system_message(self, content: str) -> None:
        message = SystemMessage(content=content)
        self.messages.append(message)
        logger.debug(f"System message added: {content}")

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