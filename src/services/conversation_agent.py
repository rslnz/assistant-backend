from typing import Any, Dict, List, Literal, TypedDict

from langchain.agents import OpenAIFunctionsAgent
from langchain_core.messages import AIMessage, BaseMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor

from src.models.chat_models import ConversationContext, Role
from src.models.tags import ResponseType
from src.services.llm_service import LLMService
from src.services.message_preparer import MessagePreparer
from src.tools.math import math_evaluation
from src.tools.web_parse import web_parse
from src.tools.web_search import web_search


class AgentState(TypedDict):
    messages: List[BaseMessage]

class ConversationManager:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.message_preparer = MessagePreparer()
        self.tools = [math_evaluation, web_parse, web_search]
        self.agent = self._create_agent()
        self.summarize_chain = self._create_summarize_chain()

    def _create_agent(self):
        tool_executor = ToolExecutor(self.tools)
        
        agent = OpenAIFunctionsAgent.from_llm_and_tools(
            llm=self.llm_service.llm,
            tools=self.tools
        )

        def should_continue(state: AgentState) -> Literal["continue", "end"]:
            return "continue" if isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].additional_kwargs.get("function_call") else "end"

        def agent_step(state: AgentState) -> AgentState:
            response = agent.plan(state["messages"])
            return {"messages": state["messages"] + [response]}

        def tool_step(state: AgentState) -> AgentState:
            last_message = state["messages"][-1]
            action = last_message.additional_kwargs["function_call"]
            action_input = action["arguments"]
            action_name = action["name"]
            output = tool_executor.execute(action_name, action_input)
            function_message = FunctionMessage(content=str(output), name=action_name)
            return {"messages": state["messages"] + [function_message]}

        workflow = StateGraph(AgentState)

        workflow.add_node("agent", agent_step)
        workflow.add_node("action", tool_step)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")

        return workflow.compile()

    async def process_message(self, message: str, system_prompt: str, context: ConversationContext):
        messages = self.message_preparer.prepare_messages(context, system_prompt, message)

        try:
            async for event in self.agent.astream({"messages": messages}):
                async for response in self._process_agent_event(event):
                    yield response

            self._update_context(context, message, event["messages"][-1].content)
            async for response in self._handle_summarization(context):
                yield response
            yield self.format_response(ResponseType.UPDATED_CONTEXT, context.model_dump())

        except Exception as e:
            yield self.format_response(ResponseType.ERROR, str(e))

    async def _process_agent_event(self, event: Dict):
        if "messages" in event:
            last_message = event["messages"][-1]
            if isinstance(last_message, AIMessage):
                if last_message.additional_kwargs.get("function_call"):
                    action = last_message.additional_kwargs["function_call"]
                    yield self.format_response(ResponseType.REASONING, last_message.content)
                    yield self.format_response(ResponseType.TOOL_START, {
                        "name": action["name"],
                        "input": action["arguments"],
                        "user_notification": f"Using {action['name']}"
                    })
                else:
                    yield self.format_response(ResponseType.TEXT, last_message.content)
            elif isinstance(last_message, FunctionMessage):
                yield self.format_response(ResponseType.TOOL_END, {
                    "name": last_message.name,
                    "output": last_message.content
                })

    def _update_context(self, context: ConversationContext, human_message: str, ai_message: str):
        context.add_message(Role.HUMAN, human_message)
        context.add_message(Role.AI, ai_message)

    async def _handle_summarization(self, context: ConversationContext):
        if context.needs_summarization():
            summary = await self.summarize_messages(context)
            context.summary = summary
            context.remove_summarized_messages()
            yield self.format_response(ResponseType.TEXT, "Conversation summarized.")

    async def summarize_messages(self, context: ConversationContext) -> str:
        messages_to_summarize = context.get_messages_to_summarize()
        text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages_to_summarize])
        return await self.summarize_chain.ainvoke({"text": text})

    def _create_summarize_chain(self):
        prompt = ChatPromptTemplate.from_template("Summarize the following conversation:\n\n{text}")
        return prompt | self.llm_service.llm

    def format_response(self, tag: ResponseType, content: Any) -> Dict[str, Any]:
        return {"type": tag.value, "content": content}
