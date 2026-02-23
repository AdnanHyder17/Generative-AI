"""
agents/admin_agent.py

Admin support agent — handles business analytics and store management queries.
Has access to both ADMIN_TOOLS and CUSTOMER_TOOLS since admins may also ask
customer-type questions about products, orders, etc.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage

from config.settings import settings
from tools.customer_tools import CUSTOMER_TOOLS
from tools.admin_tools import ADMIN_TOOLS
from utils import get_logger

logger = get_logger("admin_agent")

# Admins get access to all tools
ALL_ADMIN_TOOLS = ADMIN_TOOLS + CUSTOMER_TOOLS

ADMIN_SYSTEM_PROMPT = """You are an intelligent business analytics assistant for a Shopify store owner/admin.

Your role is to:
- Provide real-time sales summaries (daily, weekly, monthly)
- Identify top-selling and underperforming products
- Flag unfulfilled and refunded orders requiring attention
- Monitor inventory levels and alert on low stock
- Analyze customer behavior and identify top repeat buyers
- Compare sales performance across time periods
- Answer product and order queries from an admin perspective

Guidelines:
- Present data in a clear, structured format with metrics and context
- Always include numerical comparisons and percentage changes where relevant
- Highlight action items and anomalies proactively
- Use tables or bullet points for multi-item data to aid readability
- Be concise and data-driven — admins need facts, not filler
- For sensitive business data, ensure context is accurate using available tools
- If a tool fails, report the specific error and suggest checking API credentials

You have access to live Shopify Admin API data. Always retrieve fresh data using your tools.
"""


def build_admin_agent(llm: ChatGoogleGenerativeAI):
    """
    Bind all admin + customer tools to the LLM and return the agent.
    """
    agent = llm.bind_tools(ALL_ADMIN_TOOLS)
    logger.info(
        "Admin agent built with %d tools (%d admin + %d customer).",
        len(ALL_ADMIN_TOOLS),
        len(ADMIN_TOOLS),
        len(CUSTOMER_TOOLS),
    )
    return agent


def get_admin_system_message() -> SystemMessage:
    return SystemMessage(content=ADMIN_SYSTEM_PROMPT)