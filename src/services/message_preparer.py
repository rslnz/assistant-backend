import datetime
from typing import List

import pytz
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from src.models.base_models import Role
from src.models.conversation_state import ConversationState
from src.config import settings


class MessagePreparer:
    def __init__(self, tools: List[BaseTool]):
        self.tools = tools

    def prepare_messages(self, state: ConversationState) -> List[BaseMessage]:
        messages = []

        if state.system_prompt:
            messages.append(SystemMessage(content=state.system_prompt))

        if state.summary:
            messages.append(SystemMessage(content=f"Conversation summary: {state.summary}"))

        messages.extend(self._get_recent_messages(state))

        if state.user_input:
            messages.append(HumanMessage(content=state.user_input))

        messages.append(SystemMessage(content=self._get_format_instructions()))
        
        return messages

    def _get_recent_messages(self, state: ConversationState) -> List[BaseMessage]:
        message_map = {
            Role.HUMAN: HumanMessage,
            Role.AI: AIMessage,
            Role.SYSTEM: SystemMessage
        }
        recent_messages = state.get_recent_messages(limit=settings.MAX_HISTORY_MESSAGES)
        return [message_map[entry.role](content=entry.content) for entry in recent_messages]

    def _get_format_instructions(self) -> str:
        tool_instructions = "\n".join(
            f"- {tool.name}: {tool.description}\n  Arguments: {self._get_tool_args(tool)}"
            for tool in self.tools
        )
        
        current_time_utc = datetime.datetime.now(pytz.UTC)
        
        return f"""
        Current date and time information:
        - UTC: {current_time_utc.strftime("%Y-%m-%d %H:%M:%S %Z")}

        Available tools:
        {tool_instructions}

        Your response MUST include these sections in order, each starting with its tag on a new line:
        §plan, §reasoning, §text (optional), §tool (optional), §status, §summary

        Key instructions:
        1. §plan: JSON with steps, current_step, and total_steps. Update when:
           - Starting a new task/subtask
           - Completing a step
           - Modifying due to new information or errors
        2. §reasoning: JSON with thought and user_notification (1-5 words)
        3. §text: Use ONLY for clarifications or presenting results
        4. §status: JSON with status (continue/clarify/complete) and reason
        5. §summary: Brief conversation overview
        6. Use §tool only when necessary for complex tasks. Include 'user_notification' in the tool JSON (1-5 words).
        7. Handle tool errors:
           - Acknowledge error, try alternative or skip
           - Don't repeat failed tool use
           - Inform user if task can't be completed
        8. Use user's language for §text, user_notification, and status reason
        9. One step per request, don't loop indefinitely on failures

        Example: Currency conversion with fallback
        §plan
        {{"steps":[{{"description":"Get exchange rate","status":"in_progress"}},{{"description":"Calculate conversion","status":"pending"}},{{"description":"Present result","status":"pending"}}],"current_step":1,"total_steps":3}}
        §reasoning
        {{"thought":"Need current EUR to USD exchange rate","user_notification":"Checking rate"}}
        §tool
        {{"name":"web_search","arguments":{{"query":"current EUR to USD exchange rate"}},"user_notification":"Checking exchange rate"}}
        §status
        {{"status":"continue","reason":"Fetching exchange rate"}}
        §summary
        User asked to convert euros to dollars, initiating web search for current exchange rate.

        Next iteration (assuming web search failed):
        §plan
        {{"steps":[{{"description":"Get exchange rate","status":"failed"}},{{"description":"Use approximate rate","status":"in_progress"}},{{"description":"Calculate conversion","status":"pending"}},{{"description":"Present result with disclaimer","status":"pending"}}],"current_step":2,"total_steps":4}}
        §reasoning
        {{"thought":"Web search failed, using approximate rate","user_notification":"Using estimate"}}
        §status
        {{"status":"continue","reason":"Using approximate exchange rate"}}
        §summary
        Web search for exchange rate failed, proceeding with approximate rate for conversion.

        Final iteration:
        §plan
        {{"steps":[{{"description":"Get exchange rate","status":"failed"}},{{"description":"Use approximate rate","status":"completed"}},{{"description":"Calculate conversion","status":"completed"}},{{"description":"Present result with disclaimer","status":"in_progress"}}],"current_step":4,"total_steps":4}}
        §reasoning
        {{"thought":"Conversion calculated, need to present result with disclaimer","user_notification":"Showing result"}}
        §text
        Based on an approximate exchange rate (as I couldn't fetch the current rate), 100 EUR is about 110 USD. Please note this is an estimate and the actual rate may vary. For accurate conversions, I recommend checking with a bank or financial website.
        §status
        {{"status":"complete","reason":"Conversion estimate provided"}}
        §summary
        Provided estimated EUR to USD conversion with disclaimer due to inability to fetch current exchange rate.

        Failure to follow this format will result in an invalid response.
        """

    def _get_tool_args(self, tool: BaseTool) -> str:
        if hasattr(tool, 'args_schema') and tool.args_schema is not None:
            return ', '.join(f"{name}: {field.description}" for name, field in tool.args_schema.model_fields.items())
        return "No arguments"
