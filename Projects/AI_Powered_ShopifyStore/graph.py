"""
graph.py — Main LangGraph StateGraph for Silk Skin AI Agent System.

Architecture:
─────────────────────────────────────────────────────────────────
                        ┌──────────────────────────────┐
                        │          START               │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │          router              │
                        │  (determines target agent    │
                        │   based on user_role)        │
                        └────────┬─────────────────────┘
                                 │
              ┌──────────────────┴────────────────────┐
              │                                        │
              ▼                                        ▼
 ┌────────────────────────┐          ┌─────────────────────────────┐
 │  customer_support_agent │          │   admin_support_agent        │
 │  (Aria)                 │          │   (Atlas)                    │
 │  - search_products      │          │   - get_revenue_summary      │
 │  - get_best_sellers     │          │   - get_top_products         │
 │  - get_order_status     │          │   - get_unfulfilled_orders   │
 │  - get_store_policies   │          │   - get_low_inventory        │
 └────────────┬────────────┘          │   - compare_sales_periods   │
              │                       │   - get_refunded_orders      │
              └──────────┬────────────│   - get_zero_sales_products  │
                         │            │   - get_recent_orders        │
                         │            │   + all customer tools       │
                         │            └──────────┬──────────────────┘
                         │                       │
                         └──────────┬────────────┘
                                    │
                                    ▼
                        ┌──────────────────────────────┐
                        │          END                 │
                        └──────────────────────────────┘

Routing Rules:
- user_role == "customer"  → ALWAYS goes to customer_support_agent
- user_role == "admin"     → goes to admin_support_agent
  (Admin can still trigger customer tools via admin_support_agent's tool set)
─────────────────────────────────────────────────────────────────

Memory: InMemorySaver (conversation memory across turns per thread_id)
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

from state import State, Context
from agents import create_customer_agent, create_admin_agent


# ─────────────────────────────────────────────
# Initialize Agents (done once at module load)
# ─────────────────────────────────────────────

_customer_agent = create_customer_agent()
_admin_agent = create_admin_agent()


# ─────────────────────────────────────────────
# Node: Router
# ─────────────────────────────────────────────

def router(state: State) -> dict:
    """
    Routing node that determines which agent handles the request.

    Rules:
    - Customers ALWAYS go to customer_support_agent.
    - Admins go to admin_support_agent.
    
    This is a pass-through node — it doesn't modify messages,
    only sets the active_agent field for the conditional edge.
    """
    role = state.get("user_role", "customer")
    if role == "admin":
        return {"active_agent": "admin_support_agent"}
    else:
        return {"active_agent": "customer_support_agent"}


def route_decision(state: State) -> str:
    """
    Conditional edge function: returns the name of the next node
    based on the active_agent field set by the router.
    """
    return state.get("active_agent", "customer_support_agent")


# ─────────────────────────────────────────────
# Node: Customer Support Agent
# ─────────────────────────────────────────────

def customer_support_node(state: State) -> dict:
    """
    Invokes the customer support agent (Aria) with the current conversation state.
    
    Passes the full message history to maintain context across turns.
    Returns updated messages including the agent's response and any tool calls.
    """
    result = _customer_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


# ─────────────────────────────────────────────
# Node: Admin Support Agent
# ─────────────────────────────────────────────

def admin_support_node(state: State) -> dict:
    """
    Invokes the admin support agent (Atlas) with the current conversation state.
    
    Admin agent has access to both admin analytics tools and customer tools,
    so it can answer any type of query an admin may have.
    
    Returns updated messages including the agent's response and any tool calls.
    """
    result = _admin_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


# ─────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the Silk Skin multi-agent LangGraph.

    Graph topology:
        START → router → [customer_support_agent | admin_support_agent] → END

    Memory:
        InMemorySaver enables multi-turn conversation memory.
        Use a consistent thread_id when invoking to maintain history.

    Returns:
        Compiled LangGraph ready for invocation.
    """
    # Initialize the graph with State and Context schemas
    graph_builder = StateGraph(
        state_schema=State,
        config_schema=Context,
    )

    # Register nodes
    graph_builder.add_node("router", router)
    graph_builder.add_node("customer_support_agent", customer_support_node)
    graph_builder.add_node("admin_support_agent", admin_support_node)

    # Define edges
    graph_builder.add_edge(START, "router")

    # Conditional routing from router to appropriate agent
    graph_builder.add_conditional_edges(
        "router",
        route_decision,
        {
            "customer_support_agent": "customer_support_agent",
            "admin_support_agent": "admin_support_agent",
        },
    )

    # Both agents end the graph after responding
    graph_builder.add_edge("customer_support_agent", END)
    graph_builder.add_edge("admin_support_agent", END)

    # Compile with in-memory checkpointing for conversation memory
    memory = InMemorySaver()
    compiled_graph = graph_builder.compile(checkpointer=memory)

    return compiled_graph


# ─────────────────────────────────────────────
# Module-level compiled graph (singleton)
# ─────────────────────────────────────────────

graph = build_graph()