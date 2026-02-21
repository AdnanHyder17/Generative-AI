# Shopify AI Agent — LangGraph Multi-Agent System

An AI-powered multi-agent system built with **LangGraph** and **Google Gemini** that provides intelligent support for both customers and store admins via live Shopify data.

---

## Architecture

```
main.py
│
├── graph/agent_graph.py          ← StateGraph + routing logic
│   ├── router_node               ← Enforces role-based access
│   ├── customer_support_agent    ← Customer queries only
│   ├── customer_tools (ToolNode) ← Executes customer tools
│   ├── admin_support_agent       ← Admin + customer queries
│   └── admin_tools (ToolNode)    ← Executes all tools
│
├── agents/
│   ├── customer_agent.py         ← LLM + CUSTOMER_TOOLS bound
│   └── admin_agent.py            ← LLM + ALL_TOOLS bound
│
├── tools/
│   ├── customer_tools.py         ← 10 customer-facing tools
│   └── admin_tools.py            ← 10 admin analytics tools
│
├── config/settings.py            ← Env vars + Shopify config
├── utils.py                      ← ShopifyClient, helpers, formatters
└── main.py                       ← CLI entry point
```

## Graph Flow

```
START → router → [customer_support_agent | admin_support_agent]
                       ↕                          ↕
              customer_tools              admin_tools
                       ↕                          ↕
              customer_support_agent    admin_support_agent
                       ↓                          ↓
                      END                        END
```

**Key routing rule:** `user_role=customer` → ALWAYS routed to `customer_support_agent`.  
`user_role=admin` → routed to `admin_support_agent` (which also has customer tools).

---

## Setup

### 1. Clone & Install

```bash
cd shopify_agent
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key
X_SHOPIFY_ACCESS_TOKEN=your_shopify_access_token
SHOPIFY_STORE_DOMAIN=your-store.myshopify.com
```

### 3. Run

```bash
# Interactive customer chat
python main.py

# Interactive admin chat
python main.py --role admin

# Demo all customer prompts
python main.py --demo

# Demo all admin prompts
python main.py --demo --role admin
```

---

## Customer Tools

| Tool | Description |
|------|-------------|
| `search_products_by_keyword` | Full-text product search |
| `filter_products_by_max_price` | Products under a price threshold |
| `check_product_availability` | Variant availability by size/color |
| `get_best_selling_products` | Top products by sales volume |
| `get_similar_products` | Tag/type-based recommendations |
| `track_order_by_id` | Order status and tracking info |
| `get_shipping_policy` | Shipping times and rates |
| `get_return_and_refund_policy` | Returns policy |
| `get_discount_and_promo_info` | Active promo codes |
| `report_damaged_item` | Damage claim guidance |

## Admin Tools (+ all Customer Tools)

| Tool | Description |
|------|-------------|
| `get_todays_sales_summary` | Today's revenue and order count |
| `get_sales_last_n_days` | Daily sales breakdown |
| `compare_this_month_vs_last_month` | MoM revenue comparison |
| `get_average_order_value_this_month` | AOV for current month |
| `get_top_selling_products_this_month` | Best sellers this month |
| `get_low_inventory_products` | Items below stock threshold |
| `get_unsold_products_last_30_days` | Stale/unsold active products |
| `get_unfulfilled_orders_count` | Open unfulfilled orders |
| `get_refunded_orders_this_week` | Refunds issued this week |
| `get_top_repeat_customers` | Highest-value repeat buyers |

---

## Sample Prompts

### Customer
- "I'm looking for summer dresses under $50."
- "Do you have this product available in size medium?"
- "Where is my order #45821?"
- "I received a damaged item. What should I do?"

### Admin
- "Show me today's total sales and number of orders."
- "Which products are low in inventory?"
- "Compare this month's sales with last month's."
- "Who are my top repeat customers?"

---

## State Schema

```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # Full conversation
    active_agent: str  # "customer" | "admin"

class Context(TypedDict):
    user_role: str     # "customer" | "admin" — determines routing
    session_id: str    # For logging
```

Memory is persisted with `InMemorySaver` — each thread maintains its own conversation history across multiple turns.

---

## Shopify API Scopes Required

Ensure your `X-Shopify-Access-Token` has these scopes:
- `read_products`
- `read_orders`
- `read_customers`
- `read_inventory`
- `read_price_rules`