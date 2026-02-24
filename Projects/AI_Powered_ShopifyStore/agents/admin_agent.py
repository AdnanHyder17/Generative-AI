"""
agents/admin_agent.py

Admin support agent — handles business analytics and store management queries.
Has access to both ADMIN_TOOLS and CUSTOMER_TOOLS since admins may also ask
customer-type questions about products, orders, etc.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage

from tools.customer_tools import CUSTOMER_TOOLS
from tools.admin_tools import ADMIN_TOOLS
from utils import get_logger

logger = get_logger("admin_agent")

# Admins get all tools: 5 admin + 4 customer = 9 total
ALL_ADMIN_TOOLS = ADMIN_TOOLS + CUSTOMER_TOOLS

ADMIN_SYSTEM_PROMPT = """You are an intelligent business analytics assistant for a Shopify store owner.

Your role is to:
- Deliver real-time sales summaries (daily, weekly, monthly, month-over-month)
- Identify top-selling and underperforming products
- Flag unfulfilled and refunded orders requiring action
- Monitor inventory and alert on low or zero stock
- Analyse customer behaviour and identify top repeat buyers
- Answer product and order questions from an admin perspective

Guidelines:
- Present data in a clear, structured format with metrics and context
- Include percentage changes and comparisons wherever useful
- Be concise and data-driven — admins need actionable facts, not filler
- For sensitive business data, always retrieve fresh data using your tools
- If a tool fails, report the specific error and suggest checking API credentials or permissions

You have access to live Shopify Admin API data through your tools. Always use them.
"""


def build_admin_agent(llm: ChatGoogleGenerativeAI):
    """Bind all admin + customer tools to the LLM and return the agent."""
    agent = llm.bind_tools(ALL_ADMIN_TOOLS)
    logger.info(
        "Admin agent built with %d tools (%d admin + %d customer).",
        len(ALL_ADMIN_TOOLS), len(ADMIN_TOOLS), len(CUSTOMER_TOOLS),
    )
    return agent


def get_admin_system_message() -> SystemMessage:
    return SystemMessage(content=ADMIN_SYSTEM_PROMPT)