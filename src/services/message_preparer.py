from typing import List, Dict
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from src.models.chat_models import ConversationContext, Role
from langchain.tools import BaseTool

class MessagePreparer:
    def __init__(self, tools: List[BaseTool]):
        self.tools = tools

    def prepare_messages(self, context: ConversationContext, system_prompt: str, message: str) -> List[BaseMessage]:
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        if context.conversation_summary:
            messages.append(SystemMessage(content=f"Conversation summary: {context.conversation_summary}"))
        
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

        Please format your response as follows:
        [CONTENT]Your response here. You can include any text within this section. Note: All text must be inside the [CONTENT] tag, except for tool usage and summary sections.[/CONTENT][TOOL]{{"name": "tool_name","action": "Brief description of what the tool will do","arguments": {{"arg1": "value1","arg2": "value2"}}}}[/TOOL][SUMMARY]Brief summary of the conversation, including main points and conclusions.[/SUMMARY]

        Important instructions for tool usage and response formatting:
        1. Actively use the provided tools to enhance your responses. If a question requires information that you're not certain about or that might be time-sensitive, use an appropriate tool to fetch accurate and up-to-date information.
        2. You can and should use multiple tools in one response if necessary to provide a comprehensive answer.
        3. Always explain your reasoning for using each tool in the [CONTENT] section before the [TOOL] section.
        4. In the "action" field of the [TOOL] section, provide a brief, user-friendly description of what the tool will do (e.g., "Searching for current weather information in Moscow").
        5. After using a tool or multiple tools, always provide a [CONTENT] section to interpret or explain the results.
        6. Consider the logical order of tool usage. For example, you might need to search for information before you can analyze it, but this should be done in separate responses.
        7. Always include a [SUMMARY] section at the end of your response, summarizing the main points of the conversation so far, including any information obtained from tools.
        8. Do not include any newlines between closing and opening tags.
        9. If you're unsure about something or need more information to provide an accurate response, don't hesitate to use a tool to find the necessary information.
        10. Remember that using tools to provide accurate, up-to-date information is preferred over relying solely on your pre-existing knowledge.
        11. IMPORTANT: Always include closing tags for all sections. Specifically, make sure to include [/CONTENT], [/TOOL], and [/SUMMARY] tags to close their respective sections.
        12. The [SUMMARY] section must always be the last section in your response.

        Remember to consider the context, previous tool results, and the conversation summary when deciding which tools to use next and how to respond. Your goal is to provide the most helpful, accurate, and up-to-date information possible, making full use of the available tools.
        """

    def _get_tool_args(self, tool: BaseTool) -> str:
        if hasattr(tool, 'args'):
            return ', '.join(f"{arg_name}: {arg_description}" for arg_name, arg_description in tool.args.items())
        return "No arguments"
