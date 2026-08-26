# agent_workflow.py
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from tools import tools

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.1
)
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: State):
    res = llm_with_tools.invoke(state["messages"])
    return {"messages": [res]}


def should_continue(state: State):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END


workflow = StateGraph(State)
workflow.add_node("agent", agent_node)
#使用官方ToolNode，不要自己写lambda调用tools.invoke
tool_node = ToolNode(tools=tools)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent")

graph = workflow.compile()

if __name__ == "__main__":
    print("Agent workflow compile success")
