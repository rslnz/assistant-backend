from typing import List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from src.models.chat_models import ConversationContext, Role

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
        
        return f"""Available tools:
        {tool_instructions}

        Format your response as follows:
        [CONTENT]Your response here. All text must be inside this tag, except for tool usage and summary.[/CONTENT]
        [TOOL]{{"name": "tool_name","action": "Brief description of tool action","arguments": {{"arg1": "value1","arg2": "value2"}}}}[/TOOL]
        [SUMMARY]Brief summary of the conversation, including main points and conclusions.[/SUMMARY]

        Key instructions:
        1. Use tools for up-to-date information when needed.
        2. Use multiple tools if necessary for comprehensive answers.
        3. Explain tool usage reasoning in [CONTENT] before [TOOL].
        4. Provide user-friendly action descriptions in [TOOL].
        5. Interpret tool results in a [CONTENT] section after tool use.
        6. Consider logical order of tool usage.
        7. Always include a [SUMMARY] at the end.
        8. No newlines between tags.
        9. Use tools when unsure or needing current information.
        10. Prefer tool-provided information over pre-existing knowledge.
        11. Always include closing tags for all sections.
        12. [SUMMARY] must be the last section.
        13. If a tool returns an error:
            - Do not retry the same tool with similar arguments.
            - Explain the error to the user in simple terms.
            - Suggest alternative approaches or ask for clarification if needed.
            - If the error seems critical (e.g., network issues, parsing problems), avoid using that tool again for the current query.

        Consider context, previous results, and conversation summary when using tools and responding.
        Handle tool errors gracefully and inform the user about any issues that may affect the quality or completeness of the information provided.
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