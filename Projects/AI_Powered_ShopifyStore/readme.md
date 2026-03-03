# 🛍️ Silk Skin AI Agent System

A production-ready multi-agent AI system for **Silk Skin**, a luxury leather goods Shopify store. Built with **LangGraph**, **Google Gemini**, and the **Shopify Admin REST API**.

---

## 📁 Project Structure

```
silk_skin_agent/
├── main.py              # CLI entrypoint — run interactive chat or batch demo
├── graph.py             # LangGraph StateGraph — builds and compiles the agent graph
├── agents.py            # Agent creation — customer & admin agents with system prompts
├── state.py             # State & Context schema definitions
├── customer_tools.py    # Tools for the customer support agent (Aria)
├── admin_tools.py       # Tools for the admin support agent (Atlas)
├── utils.py             # Shared Shopify API utilities and helpers
├── requirements.txt     # Python dependencies
└── .env.example         # Environment variable template
```

---

## 🏗️ Architecture

```
START
  │
  ▼
router  ──────────────────────────────────┐
  │ (user_role == "customer")             │ (user_role == "admin")
  ▼                                       ▼
customer_support_agent             admin_support_agent
  (Aria)                              (Atlas)
  ├── search_products                ├── get_revenue_summary
  ├── get_best_sellers               ├── get_top_products
  ├── get_order_status               ├── get_unfulfilled_orders
  └── get_store_policies             ├── get_low_inventory_products
                                     ├── compare_sales_periods
                                     ├── get_refunded_orders
                                     ├── get_zero_sales_products
                                     ├── get_recent_orders
                                     └── + all customer tools
  │                                       │
  └──────────────┬────────────────────────┘
                 ▼
               END
```

### Key Design Decisions

- **Customers → Customer Agent only**: Enforced at the router level. No customer query can reach the admin agent.
- **Admins → Admin Agent**: Admin agent has access to ALL tools (admin + customer) since admins may need to look up products and orders too.
- **Memory**: `InMemorySaver` provides per-thread conversation memory, enabling multi-turn context.
- **Graph Schema**: `StateGraph(state_schema=State, config_schema=Context)`

---

## ⚙️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=your_gemini_api_key_here
X_SHOPIFY_ACCESS_TOKEN=your_shopify_access_token_here
SHOPIFY_STORE_URL=your-store.myshopify.com
```

### 3. Run

**Interactive customer chat:**
```bash
python main.py --role customer
```

**Interactive admin chat:**
```bash
python main.py --role admin
```

**Demo batch mode (runs all sample prompts):**
```bash
python main.py --role customer --demo
python main.py --role admin --demo
```

**With persistent thread (memory across runs):**
```bash
python main.py --role customer --thread my-session-abc123
```

---

## 🤖 Agents

### Aria — Customer Support Agent

Friendly, warm, and brand-aligned. Helps customers with:

| Query Type | Tool Used |
|---|---|
| Browse by category/tag | `search_products(tag=...)` |
| Find by price range | `search_products(max_price=...)` |
| Find by color | `search_products(color=...)` |
| Check stock | `search_products(in_stock_only=True)` |
| Best sellers | `get_best_sellers()` |
| Order tracking | `get_order_status(order_number=...)` |
| Return/refund/discount policy | `get_store_policies()` |

### Atlas — Admin Support Agent

Direct, data-driven, professional. Helps admins with:

| Query Type | Tool Used |
|---|---|
| Revenue & order metrics | `get_revenue_summary(days=...)` |
| Top products by revenue | `get_top_products(days=..., top_n=...)` |
| Unfulfilled orders | `get_unfulfilled_orders()` |
| Low stock alerts | `get_low_inventory_products(threshold=...)` |
| Month-over-month comparison | `compare_sales_periods(...)` |
| Refunded orders | `get_refunded_orders(days=...)` |
| Zero-sales products | `get_zero_sales_products(days=...)` |
| Recent orders | `get_recent_orders(hours=...)` |
| 7-day summary | `get_revenue_summary(7)` + `get_top_products(7)` |

---

## 🏷️ Product Tags

Valid tags used for filtering (must match exactly):

| Tag | Description |
|---|---|
| `Wallet` | Men's and general wallets |
| `Ladies Wallet` | Women's wallets |
| `Card Holder` | Slim card holders |
| `Handbags` | Women's handbags |
| `Bags` | General bags |
| `Travel` | Travel bags and accessories |
| `Gifts` | Luxury gift sets |
| `Accessories` | Leather accessories |
| `featured collection` | Best-sellers / featured items |

> **Bag searches** automatically include both `Bags` and `Travel` tagged products.

---

## 💬 Sample Prompts

### Customer Prompts
```
I need a premium leather wallet under $100 that's good for everyday use.
I'm buying a birthday gift — can you suggest a luxury leather item?
I travel frequently for work. Which bags would suit business trips?
Can you show me your best-selling wallets and bags right now?
I ordered a wallet (Order #45821). Can you check where it is?
I received a damaged wallet today. How can I request a replacement?
Show me Card Holder products that are priced below $80.
Do you have any Handbags in brown color under $150?
I'm looking for a Ladies Wallet in red that's currently in stock.
```

### Admin Prompts
```
Give me today's total revenue, total orders, and average order value.
What are my top 5 best-selling products this month ranked by revenue?
How many orders are currently unfulfilled, and what's their total value?
Which products are low in inventory and need restocking soon?
Compare this month's sales performance with last month.
List all refunded orders from this week with refund amounts.
Show me orders placed in the last 24 hours with customer details.
Generate a 7-day sales performance summary.
```

---

## 🔒 Security Notes

- Admin tools are **never accessible** to customers — enforced at the graph router level.
- Access token is loaded from environment only — never hardcoded.
- All Shopify API calls use proper error handling with user-friendly messages.
- Inventory and order data is fetched live — not cached — ensuring accuracy.

---

## 🧩 Extending the System

**Add a new customer tool:**
1. Define it with `@tool` in `customer_tools.py`
2. Add it to the `CUSTOMER_TOOLS` list

**Add a new admin tool:**
1. Define it with `@tool` in `admin_tools.py`
2. Add it to the `ADMIN_TOOLS` list

**Add a new agent:**
1. Define a new node function in `graph.py`
2. Update the router logic and add conditional edges