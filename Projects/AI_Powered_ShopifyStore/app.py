import streamlit as st
import uuid
import os
from langchain_core.messages import HumanMessage, AIMessage
from graph import graph  # Import your existing graph

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY_2")
os.environ["LANGCHAIN_PROJECT"] = "shopify-agent"

# --- Page Config ---
st.set_page_config(page_title="Silk Skin AI Agent", page_icon="🛍️", layout="wide")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- Data: Prompts ---
PROMPTS = {
    "customer": [
        "I need a premium leather wallet under 2500 that's good for everyday use.\n What do you recommend?",
        "I'm buying a birthday gift — can you suggest a luxury leather item that feels special and elegant?",
        "I travel frequently for work. Which leather bags or travel accessories would be best for business trips?",
        "Can you show me your best-selling wallets and bags right now?",
        "I like this leather travel bag — can you suggest similar products I might also like?",
        "Is this leather handbag available in black, and is it currently in stock?",
        "I ordered a wallet (Order #45821). Can you check where it is and when it will arrive?",
        "I received a damaged wallet today. How can I request a replacement or refund?",
        "What is your return policy if I don't like the product after delivery?",
        "I'm confused between getting a wallet or a card holder. Which one would be better for minimal everyday carry?",
        "Do you have any products tagged as Ladies Wallet that are currently in stock?",
        "What is your return and refund policy?",
        "Do you offer any discounts or promo codes?",
        "Is this product available in black color?",
        "Show me Card Holder products that are priced below 2000.",
        "Do you have any Handbags in brown color under 15000?",
        "Show me Travel bags available in black.",
        "I'm looking for a Ladies Wallet in red color that's currently in stock."
    ],
    "admin": [
        "Give me today's total revenue, total orders, and average order value.",
        "What are my top 5 best-selling products this month ranked by revenue?",
        "How many orders are currently unfulfilled, and what's their total value?",
        "Which products are low in inventory and need restocking soon?",
        "Compare this month's sales performance with last month, including revenue and order count.",
        "Show me all products under the Travel tag and their current inventory levels.",
        "List all refunded orders from this week with refund amounts.",
        "Which products have not generated any sales in the last 30 days?",
        "Show me orders placed in the last 24 hours with customer details and order value.",
        "Generate a 7-day sales performance summary including total sales, orders, and top products."
    ]
}

# --- Sidebar: Configuration & Prompts ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    user_role = st.selectbox(
        "Select User Role",
        options=["customer", "admin"],
        format_func=lambda x: x.capitalize()
    )
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    
    # Display Role-Specific Prompts as Wrapped Text
    st.subheader(f"📋 {user_role.capitalize()} Test Prompts")
    
    for p in PROMPTS[user_role]:
        st.markdown(
            f"""
            <div style="background-color: #232b3b; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-size: 14px; border: 1px solid #d1d5db;">
                {p}
            </div>
            """, 
            unsafe_allow_html=True
        )

# --- Main Chat UI ---
st.title(f"🛍️ Silk Skin AI — {user_role.capitalize()} Mode")
st.info(f"**Current Session:** `{st.session_state.thread_id}`")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Logic
if prompt := st.chat_input("Enter your message here..."):
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call LangGraph
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    state_input = {
        "messages": [HumanMessage(content=prompt)],
        "user_role": user_role,
        "active_agent": "customer_support_agent", 
    }

    with st.chat_message("assistant"):
        with st.spinner(f"Agent ({user_role}) is processing..."):
            try:
                result = graph.invoke(state_input, config=config)
                messages = result.get("messages", [])
                
                ai_response = "I'm sorry, I couldn't generate a response."
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        content = msg.content
                        if isinstance(content, list):
                            content = " ".join(
                                b.get("text", "") for b in content 
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        ai_response = content
                        break
                
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
            except Exception as e:
                st.error(f"Error: {str(e)}")