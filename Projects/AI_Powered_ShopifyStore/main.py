"""
main.py — CLI entrypoint for the Silk Skin AI Agent System.

Usage:
    python main.py --role customer
    python main.py --role admin
    python main.py --role customer --thread my-session-123

The role determines which agent handles the conversation:
- customer: Routed to customer_support_agent (Aria)
- admin: Routed to admin_support_agent (Atlas)

Each --thread maintains separate conversation memory.
"""

import argparse
import uuid
import traceback
import os
from langchain_core.messages import HumanMessage, AIMessage

from graph import graph

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY_2")
os.environ["LANGCHAIN_PROJECT"] = "shopify-agent"


def run_chat(user_role: str, thread_id: str):
    """
    Start an interactive chat session with the Silk Skin AI agent.

    Args:
        user_role: Either 'customer' or 'admin'.
        thread_id: Unique session identifier for conversation memory.
    """
    role_label = "Customer" if user_role == "customer" else "Admin"
    agent_name = "Customer_Agent" if user_role == "customer" else "Admin_Agent"

    print("\n" + "═" * 60)
    print(f"  🛍️  Silk Skin AI Agent — {role_label} Mode")
    print(f"  Agent: {agent_name}")
    print(f"  Thread: {thread_id}")
    print("  Type 'exit' or 'quit' to end the session.")
    print("═" * 60 + "\n")

    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input(f"{role_label}: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print(f"\n{agent_name}: Thank you for shopping with Silk Skin. Goodbye! 👋\n")
            break

        # Build state for this turn
        state_input = {
            "messages": [HumanMessage(content=user_input)],
            "user_role": user_role,
            "active_agent": "customer_support_agent",  # default, overridden by router
        }

        print(f"\n{agent_name}: ", end="", flush=True)

        try:
            result = graph.invoke(state_input, config=config)

            messages = result.get("messages", [])
            ai_response = None

            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    content = msg.content

                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "")
                            for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )

                    ai_response = content
                    break

            if ai_response:
                print(ai_response)
            else:
                print("(No response generated)")

        except Exception as e:
            print(f"⚠️  Error: {str(e)}")
            print("Traceback: ", traceback.format_exc())

        print()  # spacing


def batch_demo(user_role: str, thread_id: str, prompts: list[str]):
    """
    Run a batch of demo prompts non-interactively.

    Args:
        user_role: 'customer' or 'admin'.
        thread_id: Session identifier.
        prompts: List of query strings to send.
    """
    agent_name = "Aria" if user_role == "customer" else "Atlas"
    role_label = "Customer" if user_role == "customer" else "Admin"
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "═" * 60)
    print(f"  🛍️  Silk Skin AI Agent — {role_label} DEMO Mode")
    print("═" * 60)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}] {role_label}: {prompt}")
        print(f"     {agent_name}: ", end="", flush=True)

        state_input = {
            "messages": [HumanMessage(content=prompt)],
            "user_role": user_role,
            "active_agent": "customer_support_agent",
        }

        try:
            result = graph.invoke(state_input, config=config)
            messages = result.get("messages", [])
            for msg in reversed(messages):
                content = msg.content
                if isinstance(content, list):
                    content = " ".join([b["text"] for b in content if b.get("type") == "text"])
                    print(content)
                    break
        except Exception as e:
            print(f"⚠️  Error: {str(e)}")

        print("-" * 60)


# ─────────────────────────────────────────────
# Demo Prompts
# ─────────────────────────────────────────────

CUSTOMER_DEMO_PROMPTS = [
    "I need a premium leather wallet under $100 that's good for everyday use. What do you recommend?",
    "I'm buying a birthday gift — can you suggest a luxury leather item that feels special and elegant?",
    "I travel frequently for work. Which leather bags or travel accessories would be best for business trips?",
    "Can you show me your best-selling wallets and bags right now?",
    "Is this leather handbag available in black, and is it currently in stock?",
    "I ordered a wallet (Order #45821). Can you check where it is and when it will arrive?",
    "I received a damaged wallet today. How can I request a replacement or refund?",
    "What is your return policy if I don't like the product after delivery?",
    "I'm confused between getting a wallet or a card holder. Which one would be better for minimal everyday carry?",
    "Do you have any products tagged as Ladies Wallet that are currently in stock?",
    "Do you offer any discounts or promo codes?",
    "Show me Card Holder products that are priced below $80.",
    "Do you have any Handbags in brown color under $150?",
    "Show me Travel bags available in black.",
    "I'm looking for a Ladies Wallet in red color that's currently in stock.",
]

ADMIN_DEMO_PROMPTS = [
    "Give me today's total revenue, total orders, and average order value.",
    "What are my top 5 best-selling products this month ranked by revenue?",
    "How many orders are currently unfulfilled, and what's their total value?",
    "Which products are low in inventory and need restocking soon?",
    "Compare this month's sales performance with last month, including revenue and order count.",
    "Show me all products under the Travel tag and their current inventory levels.",
    "List all refunded orders from this week with refund amounts.",
    "Which products have not generated any sales in the last 30 days?",
    "Show me orders placed in the last 24 hours with customer details and order value.",
    "Generate a 7-day sales performance summary including total sales, orders, and top products.",
]


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Silk Skin AI Agent — Customer & Admin Support System"
    )
    parser.add_argument(
        "--role",
        choices=["customer", "admin"],
        default="customer",
        help="User role: 'customer' or 'admin' (default: customer)",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo prompts in batch mode instead of interactive chat",
    )

    args = parser.parse_args()
    thread_id = str(uuid.uuid4())

    if args.demo:
        prompts = CUSTOMER_DEMO_PROMPTS if args.role == "customer" else ADMIN_DEMO_PROMPTS
        batch_demo(args.role, thread_id, prompts)
    else:
        run_chat(args.role, thread_id)