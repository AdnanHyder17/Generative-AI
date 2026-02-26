"""
admin_tools.py — Tools available exclusively to the Admin Support Agent.

All Shopify data is fetched via the Admin GraphQL API.

Tools:
1. get_revenue_summary        — Revenue, order count, AOV for any time window.
2. get_top_products           — Best-selling products ranked by revenue.
3. get_unfulfilled_orders     — Unfulfilled order count, value, and list.
4. get_low_inventory_products — Variants at or below a stock threshold.
5. compare_sales_periods      — Revenue/order comparison between two periods.
6. get_refunded_orders        — Refunded/partially refunded orders.
7. get_zero_sales_products    — Products with no paid sales in a period.
8. get_recent_orders          — Orders placed in the last N hours.
"""

from datetime import datetime, timezone, timedelta
from langchain_core.tools import tool
from utils import (
    gql,
    gql_paginated,
    summarize_order,
    format_money,
    PRODUCT_FIELDS,
    ORDER_FIELDS,
)


def _iso_range(days_back: int = 0, hours_back: int = 0) -> tuple:
    """Return (start_iso, end_iso) strings for a past time window."""
    now = datetime.now(timezone.utc)
    delta = timedelta(days=days_back, hours=hours_back)
    start = now - delta
    return start.isoformat(), now.isoformat()


def _fetch_orders_gql(query_filter: str) -> list:
    """
    Fetch all orders matching a Shopify GraphQL query filter string,
    paginated automatically.

    Args:
        query_filter: Shopify search query string, e.g.
                      'financial_status:paid AND created_at:>2026-01-01'

    Returns:
        List of raw GraphQL order nodes.
    """
    gql_query = f"""
    query ($cursor: String, $query: String) {{
        orders(first: 250, after: $cursor, query: $query) {{
            pageInfo {{ hasNextPage endCursor }}
            edges {{
                node {{
                    {ORDER_FIELDS}
                }}
            }}
        }}
    }}
    """
    return gql_paginated(gql_query, variables={"query": query_filter}, data_path=["orders"])


def _fetch_products_gql(query_filter: str = "status:active") -> list:
    """
    Fetch all products matching a query filter, paginated automatically.

    Args:
        query_filter: Shopify product search query string.

    Returns:
        List of raw GraphQL product nodes.
    """
    gql_query = f"""
    query ($cursor: String, $query: String) {{
        products(first: 250, after: $cursor, query: $query) {{
            pageInfo {{ hasNextPage endCursor }}
            edges {{
                node {{
                    {PRODUCT_FIELDS}
                }}
            }}
        }}
    }}
    """
    return gql_paginated(gql_query, variables={"query": query_filter}, data_path=["products"])


# ─────────────────────────────────────────────────────────────
# Tool 1: Revenue & Order Summary
# ─────────────────────────────────────────────────────────────

@tool
def get_revenue_summary(days: int = 1) -> dict:
    """
    Get total revenue, total order count, and average order value for a time period.

    Only counts orders with financial_status = paid.

    Use this for:
    - "What is today's revenue?" → days=1
    - "This week's sales summary" → days=7
    - "Monthly revenue" → days=30

    Args:
        days: Number of past days to include (1 = today, 7 = week, 30 = month).

    Returns:
        Dict: period_days, start_date, end_date, total_revenue (PKR),
              total_orders, average_order_value (PKR).
    """
    try:
        start_date, end_date = _iso_range(days_back=days)
        orders = _fetch_orders_gql(
            f'financial_status:paid AND created_at:>"{start_date}" AND created_at:<"{end_date}"'
        )

        total_revenue = sum(
            float(o.get("totalPriceSet", {}).get("shopMoney", {}).get("amount", 0) or 0)
            for o in orders
        )
        total_orders = len(orders)
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

        return {
            "period_days": days,
            "start_date": start_date[:10],
            "end_date": end_date[:10],
            "total_revenue": format_money(total_revenue),
            "total_orders": total_orders,
            "average_order_value": format_money(avg_order_value),
        }

    except Exception as e:
        return {"error": f"Failed to get revenue summary: {str(e)}"}


# ─────────────────────────────────────────────────────────────
# Tool 2: Top Selling Products
# ─────────────────────────────────────────────────────────────

@tool
def get_top_products(days: int = 30, top_n: int = 5) -> list:
    """
    Get the top N best-selling products ranked by total revenue in a time period.

    Use this for:
    - "Top 5 products this month"
    - "Which products are selling the most?"
    - "Revenue by product"

    Args:
        days: Number of past days to analyze (default 30).
        top_n: How many top products to return (default 5).

    Returns:
        List of dicts: {product_title, total_revenue (PKR), total_units_sold},
        ranked by revenue descending.
    """
    try:
        start_date, _ = _iso_range(days_back=days)
        orders = _fetch_orders_gql(
            f'financial_status:paid AND created_at:>"{start_date}"'
        )

        product_stats: dict = {}
        for order in orders:
            for edge in order.get("lineItems", {}).get("edges", []):
                item = edge["node"]
                title = item.get("title", "Unknown")
                qty = item.get("quantity", 0) or 0
                price = float(item.get("originalUnitPrice", 0) or 0)
                revenue = qty * price

                if title not in product_stats:
                    product_stats[title] = {"total_revenue": 0.0, "total_units_sold": 0}
                product_stats[title]["total_revenue"] += revenue
                product_stats[title]["total_units_sold"] += qty

        ranked = sorted(
            [{"product_title": k, **v} for k, v in product_stats.items()],
            key=lambda x: x["total_revenue"],
            reverse=True,
        )

        # Format revenue after sorting
        for item in ranked:
            item["total_revenue"] = format_money(item["total_revenue"])

        return ranked[:top_n]

    except Exception as e:
        return [{"error": f"Failed to get top products: {str(e)}"}]


# ─────────────────────────────────────────────────────────────
# Tool 3: Unfulfilled Orders
# ─────────────────────────────────────────────────────────────

@tool
def get_unfulfilled_orders() -> dict:
    """
    Retrieve all currently unfulfilled orders with their count and total value.

    Use this for:
    - "How many orders are unfulfilled?"
    - "What orders are pending shipment?"
    - "Show me the backlog."

    Returns:
        Dict: count, total_value (PKR), orders (list of up to 20 summarized orders).
    """
    try:
        orders = _fetch_orders_gql("fulfillment_status:unfulfilled AND status:open")

        total_value = sum(
            float(o.get("totalPriceSet", {}).get("shopMoney", {}).get("amount", 0) or 0)
            for o in orders
        )

        return {
            "count": len(orders),
            "total_value": format_money(total_value),
            "orders": [summarize_order(o) for o in orders[:20]],  # cap for LLM context
        }

    except Exception as e:
        return {"error": f"Failed to get unfulfilled orders: {str(e)}"}


# ─────────────────────────────────────────────────────────────
# Tool 4: Low Inventory Products
# ─────────────────────────────────────────────────────────────

@tool
def get_low_inventory_products(threshold: int = 3) -> list:
    """
    Find all active product variants with inventory at or below a threshold.

    Use this for:
    - "Which products need restocking?"
    - "What's low on inventory?"
    - "Show me products running out of stock."

    Args:
        threshold: Inventory level at or below which a variant is flagged (default 3).

    Returns:
        List of dicts: {product_title, variant_title, inventory_quantity, sku},
        sorted by inventory_quantity ascending (most critical first).
    """
    try:
        products = _fetch_products_gql("status:active")

        low_stock = []
        for p in products:
            product_title = p.get("title", "")
            for edge in p.get("variants", {}).get("edges", []):
                v = edge["node"]
                qty = v.get("inventoryQuantity", 0) or 0
                if qty <= threshold:
                    low_stock.append({
                        "product_title": product_title,
                        "variant_title": v.get("title", "Default"),
                        "inventory_quantity": qty,
                        "sku": v.get("sku", "N/A"),
                    })

        return sorted(low_stock, key=lambda x: x["inventory_quantity"])

    except Exception as e:
        return [{"error": f"Failed to get low inventory products: {str(e)}"}]


# ─────────────────────────────────────────────────────────────
# Tool 5: Sales Comparison (Period over Period)
# ─────────────────────────────────────────────────────────────

@tool
def compare_sales_periods(current_days: int = 30, previous_days: int = 30) -> dict:
    """
    Compare revenue and order count between the current period and a previous period.

    Use this for:
    - "Compare this month vs last month"
    - "Month-over-month performance"
    - "How did sales change?"

    Args:
        current_days: Length of the current period in days (default 30).
        previous_days: Length of the previous period in days (default 30).

    Returns:
        Dict with current_period, previous_period stats, and delta changes
        including percentage change for revenue and order count.
    """
    try:
        now = datetime.now(timezone.utc)

        current_start = (now - timedelta(days=current_days)).isoformat()
        current_end = now.isoformat()
        previous_end = (now - timedelta(days=current_days)).isoformat()
        previous_start = (now - timedelta(days=current_days + previous_days)).isoformat()

        def fetch_period_stats(start: str, end: str) -> dict:
            orders = _fetch_orders_gql(
                f'financial_status:paid AND created_at:>"{start}" AND created_at:<"{end}"'
            )
            revenue = sum(
                float(o.get("totalPriceSet", {}).get("shopMoney", {}).get("amount", 0) or 0)
                for o in orders
            )
            return {"order_count": len(orders), "revenue": revenue}

        current = fetch_period_stats(current_start, current_end)
        previous = fetch_period_stats(previous_start, previous_end)

        rev_change = current["revenue"] - previous["revenue"]
        order_change = current["order_count"] - previous["order_count"]
        rev_pct = (rev_change / previous["revenue"] * 100) if previous["revenue"] > 0 else 0
        order_pct = (order_change / previous["order_count"] * 100) if previous["order_count"] > 0 else 0

        return {
            "current_period": {
                "days": current_days,
                "start": current_start[:10],
                "end": current_end[:10],
                "revenue": format_money(current["revenue"]),
                "order_count": current["order_count"],
            },
            "previous_period": {
                "days": previous_days,
                "start": previous_start[:10],
                "end": previous_end[:10],
                "revenue": format_money(previous["revenue"]),
                "order_count": previous["order_count"],
            },
            "changes": {
                "revenue_change": format_money(rev_change),
                "revenue_change_pct": f"{rev_pct:+.1f}%",
                "order_change": order_change,
                "order_change_pct": f"{order_pct:+.1f}%",
            },
        }

    except Exception as e:
        return {"error": f"Failed to compare sales periods: {str(e)}"}


# ─────────────────────────────────────────────────────────────
# Tool 6: Refunded Orders
# ─────────────────────────────────────────────────────────────

@tool
def get_refunded_orders(days: int = 7) -> list:
    """
    Retrieve all fully or partially refunded orders from the past N days.

    Use this for:
    - "Show me refunds from this week"
    - "List all refunded orders"
    - "How much have we refunded recently?"

    Args:
        days: Number of past days to look back (default 7).

    Returns:
        List of summarized order dicts with refund transaction details.
    """
    try:
        start_date, _ = _iso_range(days_back=days)

        refunded = _fetch_orders_gql(
            f'financial_status:refunded AND created_at:>"{start_date}"'
        )
        partial = _fetch_orders_gql(
            f'financial_status:partially_refunded AND created_at:>"{start_date}"'
        )

        all_refunded = refunded + partial
        return [summarize_order(o) for o in all_refunded]

    except Exception as e:
        return [{"error": f"Failed to get refunded orders: {str(e)}"}]


# ─────────────────────────────────────────────────────────────
# Tool 7: Products with Zero Sales
# ─────────────────────────────────────────────────────────────

@tool
def get_zero_sales_products(days: int = 30) -> list:
    """
    Find active products that have generated zero paid sales in the past N days.

    Use this for:
    - "Which products aren't selling?"
    - "What products have had no sales this month?"
    - "Show me non-performing / dead stock."

    Args:
        days: Number of past days to analyze (default 30).

    Returns:
        List of product title strings with no sales in the period,
        or a confirmation message if all products have sold.
    """
    try:
        start_date, _ = _iso_range(days_back=days)

        # Get all active product titles
        products = _fetch_products_gql("status:active")
        all_titles = {p.get("title") for p in products}

        # Get titles of products that appeared in paid orders
        orders = _fetch_orders_gql(
            f'financial_status:paid AND created_at:>"{start_date}"'
        )
        sold_titles = set()
        for order in orders:
            for edge in order.get("lineItems", {}).get("edges", []):
                sold_titles.add(edge["node"].get("title"))

        zero_sales = sorted(all_titles - sold_titles)
        return zero_sales if zero_sales else ["All products have had at least one sale in this period."]

    except Exception as e:
        return [f"Error: Failed to get zero-sales products: {str(e)}"]


# ─────────────────────────────────────────────────────────────
# Tool 8: Recent Orders
# ─────────────────────────────────────────────────────────────

@tool
def get_recent_orders(hours: int = 24) -> list:
    """
    Retrieve orders placed in the last N hours with customer details and order values.

    Use this for:
    - "Show me orders from the last 24 hours"
    - "What orders came in today?"
    - "Give me today's new orders"

    Args:
        hours: How many hours back to look (default 24).

    Returns:
        List of summarized order dicts with customer info and order value.
    """
    try:
        start_date, _ = _iso_range(hours_back=hours)
        orders = _fetch_orders_gql(f'created_at:>"{start_date}"')
        return [summarize_order(o) for o in orders]

    except Exception as e:
        return [{"error": f"Failed to get recent orders: {str(e)}"}]


# ─────────────────────────────────────────────────────────────
# Exported tool list
# ─────────────────────────────────────────────────────────────

ADMIN_TOOLS = [
    get_revenue_summary,
    get_top_products,
    get_unfulfilled_orders,
    get_low_inventory_products,
    compare_sales_periods,
    get_refunded_orders,
    get_zero_sales_products,
    get_recent_orders,
]