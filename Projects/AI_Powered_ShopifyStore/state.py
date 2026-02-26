"""
state.py — LangGraph State and Context schemas for Silk Skin AI Agent.
"""

from typing import Annotated, Literal
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage


class State(MessagesState):
    """
    Shared conversation state.
    
    Inherits 'messages' (list of BaseMessage) from MessagesState with
    automatic append-mode reducer.
    
    Additional fields:
    - user_role: Whether the current user is a 'customer' or 'admin'.
    - active_agent: Which agent is currently handling the conversation.
    - last_tool_output: Optional scratch space for tool results passed between nodes.
    """
    user_role: Literal["customer", "admin"]
    active_agent: Literal["customer_support_agent", "admin_support_agent", "__end__"]


class Context:
    """
    Immutable context injected at graph compile time.
    
    Fields:
    - store_name: The name of the Shopify store.
    - store_description: Short brand description for agent personality.
    """
    store_name: str = "Silk Skin"
    store_description: str = (
        "Silk Skin is a luxury leather goods brand offering premium wallets, "
        "handbags, card holders, bags, travel accessories, and gift sets. "
        "All products are crafted from the finest leather with timeless elegance."
    )