from typing import List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from src.models.chat_models import ConversationContext, Role
from src.models.prompt_structures import LLMProcessingState

class MessagePreparer:
    def __init__(self, tools: List[BaseTool]):
        self.tools = tools

    def prepare_messages(self, context: ConversationContext, state: LLMProcessingState, message: str) -> List[BaseMessage]:
        messages = []
        
        if state.summary:
            messages.append(SystemMessage(content=f"Conversation summary: {state.summary}"))
        
        messages.extend(self._get_recent_messages(context))
        messages.append(HumanMessage(content=message))
        messages.append(SystemMessage(content=self._get_format_instructions()))
        
        return messages

    def _get_recent_messages(self, context: ConversationContext) -> List[BaseMessage]:
        message_map = {
            Role.HUMAN.value: HumanMessage,
            Role.AI.value: AIMessage,
            Role.SYSTEM.value: SystemMessage
        }
        return [message_map[entry['role']](content=entry['content']) for entry in context.get_recent_messages()]

    def _get_format_instructions(self) -> str:
        tool_instructions = "\n".join(
            f"- {tool.name}: {tool.description}\n  Arguments: {self._get_tool_args(tool)}"
            for tool in self.tools
        )
        
        return f"""
        Available tools:
        {tool_instructions}

        CRITICAL: You MUST include [PLAN], [REASONING], [STATUS], and [SUMMARY] in EVERY response.

        1. [PLAN]{{
            "steps": [
                {{
                    "description": "Step description",
                    "status": "pending|in_progress|completed|failed"
                }}
            ],
            "current_step": 1,
            "total_steps": 3
        }}[/PLAN]

        2. [REASONING]{{
            "thought": "Your detailed internal reasoning process",
            "user_notification": "Brief 1-5 word action description"
        }}[/REASONING]

        3. [TOOL]{{
            "name": "tool_name",
            "arguments": {{"arg1": "value1", "arg2": "value2"}},
            "user_notification": "Brief 1-5 word action description"
        }}[/TOOL] (Include only if a tool is being used)

        4. [TEXT]Your response to the user. This is the ONLY content the user will see directly. It can include your final answer or requests for clarification.[/TEXT]

        5. [STATUS]{{
            "status": "continue|clarify|complete",
            "reason": "Brief explanation for the current status"
        }}[/STATUS]

        6. [SUMMARY]Brief summary of the entire conversation, including main points and conclusions.[/SUMMARY]

        Key instructions:
        1. Use [TOOL] when you need to gather information or perform an action.
        2. Use [TEXT] ONLY for direct communication with the user. This includes:
           - Final answers to their questions
           - Requests for clarification or additional information
           - Updates on the progress of complex tasks
        3. Keep "user_notification" in [REASONING] and [TOOL] to 1-5 words.
        4. Be conversational and natural in [TEXT], mentioning actions casually if needed.
        5. For [STATUS]:
           - Use 'continue' if more steps are needed.
           - Use 'clarify' if you need user input.
           - Use 'complete' when the task is finished.
        6. Update the [PLAN] as you progress, changing step statuses and current_step.

        Example response structure:
        [PLAN]{{"steps":[{{"description":"Gather information about the topic","status":"completed"}},{{"description":"Analyze and summarize the information","status":"in_progress"}}],"current_step":2,"total_steps":2}}[/PLAN]
        [REASONING]{{"thought":"Need to analyze the gathered information","user_notification":"Analyzing data"}}[/REASONING]
        [TOOL]{{"name":"web_parse","arguments":{{"url":"https://example.com"}},"user_notification":"Parsing webpage"}}[/TOOL]
        [TEXT]I've found some information about your question, but I need a bit more clarity. Could you please specify which aspect of the topic you're most interested in?[/TEXT]
        [STATUS]{{"status":"clarify","reason":"Need more specific information from user"}}[/STATUS]
        [SUMMARY]User asked about X. Gathered general information, now seeking clarification for more focused analysis.[/SUMMARY]

        Remember:
        - Always include ALL required tags ([PLAN], [REASONING], [TEXT], [STATUS], [SUMMARY]) in EVERY response.
        - [TEXT] is the ONLY content the user sees directly. Use it for answers, clarifications, and progress updates.
        - Put each tag on a separate line.
        - Do not use line breaks within tags.
        """

    def _get_tool_args(self, tool: BaseTool) -> str:
        if hasattr(tool, 'args_schema') and tool.args_schema is not None:
            return ', '.join(f"{name}: {field.description}" for name, field in tool.args_schema.model_fields.items())
        return "No arguments"

    def get_tool_error_message(self, tool_name: str, error: str) -> str:
        base_message = (
            f"Error using {tool_name}: {error}. "
            f"Do not use this tool again for this query. "
        )
        
        additional_info = (
            f"This error might affect the quality or completeness of the information provided. "
            f"Consider alternative approaches or ask the user for different ways to obtain the required information. "
            f"If you need this type of information, try rephrasing the query or using a different method."
        )
        
        return base_message + additional_info