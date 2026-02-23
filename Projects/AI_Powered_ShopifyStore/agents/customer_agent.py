"""
agents/customer_agent.py

Customer support agent — handles all customer-facing queries.
Uses the Gemini model bound to CUSTOMER_TOOLS only.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage

from config.settings import settings
from tools.customer_tools import CUSTOMER_TOOLS
from utils import get_logger

logger = get_logger("customer_agent")

CUSTOMER_SYSTEM_PROMPT = """You are a friendly, helpful customer support assistant for an online Shopify store.

Your role is to:
- Help customers find products they're looking for
- Check product availability by size, color, or variant
- Recommend best-selling and similar products
- Track customer orders by order ID
- Explain shipping times and policies
- Share return and refund policy
- Help with damaged or defective items
- Share current discounts and promotions

Guidelines:
- Always be warm, empathetic, and solution-oriented
- If you need an order ID, ask the customer politely
- Format product recommendations in a clear, readable way
- For damaged items, always express sympathy first
- Never make up information — use the tools provided
- If a tool fails, apologize and suggest the customer contact support directly

You have access to live Shopify store data through your tools. Always use the tools to fetch accurate, up-to-date information rather than guessing.
"""


def build_customer_agent(llm: ChatGoogleGenerativeAI):
    """
    Bind the customer tools to the LLM and return a runnable agent.
    The agent is a simple tool-bound model; tool execution is handled by the graph.
    """
    agent = llm.bind_tools(CUSTOMER_TOOLS)
    logger.info("Customer agent built with %d tools.", len(CUSTOMER_TOOLS))
    return agent


def get_customer_system_message() -> SystemMessage:
    return SystemMessage(content=CUSTOMER_SYSTEM_PROMPT)