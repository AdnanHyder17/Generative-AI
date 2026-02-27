"""
main.py — CLI entrypoint for the Silk Skin AI Agent System.

Usage:
    python main.py --role customer
    python main.py --role admin

The role determines which agent handles the conversation:
- customer: Routed to customer_support_agent
- admin: Routed to admin_support_agent

Each --thread maintains separate conversation memory.
"""

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY_2")
os.environ["LANGCHAIN_PROJECT"] = "shopify-agent"

import argparse
import uuid
import traceback
import os
from langchain_core.messages import HumanMessage, AIMessage

from graph import graph


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

    args = parser.parse_args()
    thread_id = str(uuid.uuid4())

    run_chat(args.role, thread_id)