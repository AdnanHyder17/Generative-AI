"""
main.py - Entry point for the Shopify AI Agent system.

Run modes:
  python main.py                    → interactive chat (customer by default)
  python main.py --role admin       → interactive chat as admin
  python main.py --demo             → run all sample prompts and print results
  python main.py --demo --role admin → run admin demo prompts
"""

import argparse
import sys
import uuid
from langchain_core.messages import HumanMessage

from graph.agent_graph import build_graph
from utils import get_logger

logger = get_logger("main")

# ─── Sample Prompts ───────────────────────────────────────────────────────────

CUSTOMER_DEMO_PROMPTS = [
    "I'm looking for summer dresses under $50.",
    "Do you have this product available in size medium?",
    "Can you recommend best-selling products right now?",
    "Where is my order #45821?",
    "How long does shipping take to California?",
    "What is your return and refund policy?",
    "Do you offer any discounts or promo codes?",
    "Is this product available in black color?",
    "Can you suggest products similar to this one?",
    "I received a damaged item. What should I do?",
]

ADMIN_DEMO_PROMPTS = [
    "Show me today's total sales and number of orders.",
    "What are my top 5 selling products this month?",
    "How many orders are currently unfulfilled?",
    "Which products are low in inventory?",
    "Show me sales performance for the last 7 days.",
    "Who are my top repeat customers?",
    "What is the average order value this month?",
    "List all refunded orders from this week.",
    "Which products have not sold in the last 30 days?",
    "Compare this month's sales with last month's sales.",
]


# ─── Chat Runner ──────────────────────────────────────────────────────────────

def run_query(
    graph,
    user_input: str,
    role: str,
    thread_id: str,
) -> str:
    """
    Send a single user message to the compiled graph and return the AI response.

    Args:
        graph: Compiled LangGraph
        user_input: The user's message text
        role: "customer" or "admin"
        thread_id: Unique thread ID for memory persistence

    Returns:
        The assistant's response text
    """
    config = {
        "configurable": {
            "user_role": role,
            "session_id": thread_id,
            "thread_id": thread_id,  # Required by InMemorySaver
        }
    }

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        messages = result.get("messages", [])
        # Get last AI message
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                return msg.content
        return "No response generated."
    except Exception as e:
        logger.error("Graph invocation error: %s", e)
        return f"System error: {e}"


def run_demo(graph, role: str) -> None:
    """Run all demo prompts for a given role and print results."""
    prompts = ADMIN_DEMO_PROMPTS if role == "admin" else CUSTOMER_DEMO_PROMPTS
    thread_id = f"demo-{role}-{uuid.uuid4().hex[:8]}"

    print(f"\n{'=' * 60}")
    print(f"  DEMO MODE — Role: {role.upper()}")
    print(f"{'=' * 60}\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] USER: {prompt}")
        print("-" * 50)
        response = run_query(graph, prompt, role, thread_id)
        print(f"ASSISTANT:\n{response}")
        print("=" * 60 + "\n")


def run_interactive(graph, role: str) -> None:
    """Start an interactive chat session."""
    thread_id = f"session-{role}-{uuid.uuid4().hex[:8]}"

    print(f"\n{'=' * 60}")
    print(f"  Shopify AI Agent — {'Customer' if role == 'customer' else 'Admin'} Mode")
    print(f"  Thread ID: {thread_id}")
    print(f"  Type 'exit' or 'quit' to end the session.")
    print(f"{'=' * 60}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        response = run_query(graph, user_input, role, thread_id)
        print(f"\nAssistant: {response}\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Shopify AI Agent — LangGraph Multi-Agent System"
    )
    parser.add_argument(
        "--role",
        choices=["customer", "admin"],
        default="customer",
        help="User role for this session (default: customer)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo prompts instead of interactive chat",
    )
    args = parser.parse_args()

    logger.info("Initializing Shopify AI Agent (role=%s, demo=%s)...", args.role, args.demo)

    try:
        graph, _ = build_graph()
    except EnvironmentError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("Please ensure your .env file is set up correctly (see .env.example).")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Failed to initialize graph: {e}")
        sys.exit(1)

    if args.demo:
        run_demo(graph, args.role)
    else:
        run_interactive(graph, args.role)


if __name__ == "__main__":
    main()