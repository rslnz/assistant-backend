from typing import List, Dict, Any, Optional
from pydantic import Field
from src.models.base_models import BaseContext, Role, MessageEntry
from src.models.chat_models import ConversationContext
from src.models.prompt_structures import Plan, Status

class ConversationState(BaseContext):
    system_prompt: str = ""
    user_input: str = ""
    current_plan: Optional[Plan] = None
    reasoning_history: List[str] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    status: Optional[Status] = None
    max_iterations: int = 3

    @classmethod
    def from_context(cls, context: Optional['ConversationContext'], system_prompt: str, user_input: str) -> 'ConversationState':
        if context is None:
            return cls(system_prompt=system_prompt, user_input=user_input)
        return cls(history=context.history, summary=context.summary, system_prompt=system_prompt, user_input=user_input)

    def add_message(self, role: Role, content: str) -> None:
        self.history.append(MessageEntry(role=role, content=content))

    def add_human_message(self, message: str) -> None:
        self.add_message(Role.HUMAN, message)

    def add_ai_message(self, message: str) -> None:
        self.add_message(Role.AI, message)

    def add_system_message(self, message: str) -> None:
        self.add_message(Role.SYSTEM, message)

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def update_plan(self, plan: Plan) -> None:
        self.current_plan = plan

    def add_reasoning(self, reasoning: str) -> None:
        self.reasoning_history.append(reasoning)

    def add_tool_result(self, result: Dict[str, Any]) -> None:
        self.tool_results.append(result)

    def update_summary(self, summary: str) -> None:
        self.summary = summary

    def update_status(self, status: Status) -> None:
        self.status = status

    def prepare_for_next_iteration(self) -> None:
        tool_results_summary = []
        error_message_template = (
            "Error using {tool_name}: {error}. "
            "Do not use this tool again for this query. "
            "Consider alternative approaches or rephrase the query."
        )

        for result in self.tool_results:
            if 'error' in result:
                error_message = error_message_template.format(
                    tool_name=result['name'],
                    error=result['error']
                )
                tool_results_summary.append(error_message)
            else:
                tool_results_summary.append(f"Tool {result['name']} result: {result['result']}")

        self.tool_results = []
        
        current_step = self.current_plan.current_step if self.current_plan else "Unknown"
        total_steps = self.current_plan.total_steps if self.current_plan else "Unknown"

        continuation_message = (
            f"Current progress: Step {current_step} of {total_steps}\n\n"
            f"Continue with the next step of your plan. Here's a summary of the current state:\n\n"
            f"Current plan: {self.current_plan.model_dump_json() if self.current_plan else 'No plan set'}\n\n"
            f"Previous reasoning steps:\n"
            f"{' '.join(self.reasoning_history)}\n\n"
            f"Recent tool results:\n"
            f"{' | '.join(tool_results_summary)}\n\n"
            f"Based on this information, proceed with the next step of the plan or adjust it if necessary. "
            f"If you need more information, consider using available tools, but avoid repeating failed tool calls. "
            f"If a tool failed, try alternative approaches or rephrase the query."
        )

        self.add_system_message(continuation_message)

    def get_updated_context(self) -> 'ConversationContext':
        cleaned_history = self._clean_history()
        return ConversationContext(history=cleaned_history, summary=self.summary)

    def get_recent_messages(self, limit: int) -> List[MessageEntry]:
        non_system_messages = [msg for msg in self.history if msg.role != Role.SYSTEM]
        return non_system_messages[-limit:]

    def _clean_history(self) -> List[MessageEntry]:
        return [msg for msg in self.history if msg.role != Role.SYSTEM]