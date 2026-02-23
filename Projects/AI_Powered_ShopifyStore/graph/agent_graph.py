"""
graph/agent_graph.py

Builds the LangGraph StateGraph with:
  - state_schema  : tracks messages + active agent
  - context_schema: carries user_role and session config (injected at runtime)
  - customer_support_agent node
  - admin_support_agent node
  - router node that enforces role-based access (customers → customer agent only)
  - tool execution nodes (ToolNode per agent)
  - InMemorySaver checkpointer for persistent memory
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from config.settings import settings
from agents.customer_agent import build_customer_agent, get_customer_system_message
from agents.admin_agent import build_admin_agent, get_admin_system_message
from tools.customer_tools import CUSTOMER_TOOLS
from tools.admin_tools import ADMIN_TOOLS
from utils import get_logger

logger = get_logger("agent_graph")


# ─── Schemas ─────────────────────────────────────────────────────────────────

class State(TypedDict):
    """
    Mutable conversation state passed between nodes.

    messages   : Full conversation history (auto-merged by add_messages reducer)
    active_agent: Which agent is currently handling the request
    """
    messages: Annotated[list[BaseMessage], add_messages]
    active_agent: str  # "customer" | "admin"


class Context(TypedDict):
    """
    Immutable context injected at graph invocation time.

    user_role  : "customer" | "admin" — determines routing
    session_id : Optional session identifier for logging
    """
    user_role: str
    session_id: str


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_graph() -> tuple:
    """
    Construct and compile the multi-agent graph.

    Returns:
        (compiled_graph, checkpointer)
    """
    settings.validate()

    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=1.0,
    )

    customer_agent = build_customer_agent(llm)
    admin_agent = build_admin_agent(llm)

    all_admin_tools = ADMIN_TOOLS + CUSTOMER_TOOLS

    customer_tool_node = ToolNode(CUSTOMER_TOOLS)
    admin_tool_node = ToolNode(all_admin_tools)

    # ── Node: Router ──────────────────────────────────────────────────────────
    def router_node(state: State, config: dict) -> dict:
        """
        Inspect the configurable context to determine user role and set active_agent.
        Customers are ALWAYS routed to the customer agent.
        Admins start at the admin agent (they can also ask customer questions).
        """
        ctx: Context = config.get("configurable", {})
        user_role = ctx.get("user_role", "customer").lower()

        if user_role == "admin":
            agent = "admin"
            system_msg = get_admin_system_message()
        else:
            agent = "customer"
            system_msg = get_customer_system_message()

        # Prepend system message if not already present
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [system_msg] + list(messages)

        logger.info("Router: user_role=%s → active_agent=%s", user_role, agent)
        return {"messages": messages, "active_agent": agent}

    # ── Node: Customer Agent ──────────────────────────────────────────────────
    def customer_agent_node(state: State) -> dict:
        """Run the customer support agent on the current message history."""
        try:
            response = customer_agent.invoke(state["messages"])
            logger.info("Customer agent responded (tool_calls=%d).",
                        len(getattr(response, "tool_calls", []) or []))
            return {"messages": [response], "active_agent": "customer"}
        except Exception as e:
            logger.error("Customer agent error: %s", e)
            error_msg = AIMessage(
                content=(
                    "I apologize, I'm having trouble processing your request right now. "
                    "Please try again or contact our support team directly."
                )
            )
            return {"messages": [error_msg], "active_agent": "customer"}

    # ── Node: Admin Agent ─────────────────────────────────────────────────────
    def admin_agent_node(state: State) -> dict:
        """Run the admin support agent on the current message history."""
        try:
            response = admin_agent.invoke(state["messages"])
            logger.info("Admin agent responded (tool_calls=%d).",
                        len(getattr(response, "tool_calls", []) or []))
            return {"messages": [response], "active_agent": "admin"}
        except Exception as e:
            logger.error("Admin agent error: %s", e)
            error_msg = AIMessage(
                content=(
                    f"Error processing admin request: {e}. "
                    "Please verify your Shopify API credentials and try again."
                )
            )
            return {"messages": [error_msg], "active_agent": "admin"}

    # ── Edge Conditions ───────────────────────────────────────────────────────

    def after_router(state: State) -> Literal["customer_support_agent", "admin_support_agent"]:
        """Route to the correct agent after the router sets active_agent."""
        return (
            "admin_support_agent"
            if state.get("active_agent") == "admin"
            else "customer_support_agent"
        )

    def after_customer_agent(state: State) -> Literal["customer_tools", "__end__"]:
        """If the customer agent made tool calls, run tools; otherwise end."""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "customer_tools"
        return "__end__"

    def after_admin_agent(state: State) -> Literal["admin_tools", "__end__"]:
        """If the admin agent made tool calls, run tools; otherwise end."""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "admin_tools"
        return "__end__"

    def after_customer_tools(state: State) -> Literal["customer_support_agent"]:
        """After customer tools execute, loop back to customer agent."""
        return "customer_support_agent"

    def after_admin_tools(state: State) -> Literal["admin_support_agent"]:
        """After admin tools execute, loop back to admin agent."""
        return "admin_support_agent"

    # ── Graph Assembly ────────────────────────────────────────────────────────

    checkpointer = InMemorySaver()

    graph = StateGraph(state_schema=State, config_schema=Context)

    graph.add_node("router", router_node)
    graph.add_node("customer_support_agent", customer_agent_node)
    graph.add_node("admin_support_agent", admin_agent_node)
    graph.add_node("customer_tools", customer_tool_node)
    graph.add_node("admin_tools", admin_tool_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", after_router)

    graph.add_conditional_edges(
        "customer_support_agent",
        after_customer_agent,
        {"customer_tools": "customer_tools", "__end__": END},
    )
    graph.add_conditional_edges(
        "admin_support_agent",
        after_admin_agent,
        {"admin_tools": "admin_tools", "__end__": END},
    )

    graph.add_conditional_edges(
        "customer_tools",
        after_customer_tools,
        {"customer_support_agent": "customer_support_agent"},
    )
    graph.add_conditional_edges(
        "admin_tools",
        after_admin_tools,
        {"admin_support_agent": "admin_support_agent"},
    )

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Graph compiled successfully.")
    return compiled, checkpointer