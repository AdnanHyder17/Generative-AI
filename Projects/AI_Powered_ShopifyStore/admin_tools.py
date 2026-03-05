"""
admin_tools.py — Tools for the Admin Support Agent (Shopify Admin GraphQL API).

Tools:
    fetch_today_date           — Current date for dynamic queries.
    get_revenue_summary        — Revenue, order count, AOV, and top products for a time window.
    get_unfulfilled_orders     — Unfulfilled order count, value, and list.
    get_low_inventory_products — Variants at or below a stock threshold.
    compare_sales_periods      — Revenue/order comparison between two periods.
    get_refunded_orders        — Refunded/partially refunded orders.
    get_zero_sales_products    — Products with no paid sales in a period.
    get_recent_orders          — Orders placed in a given date range.
"""

from datetime import date
from langchain.tools import tool
from rapidfuzz import process, fuzz
from utils import gql_paginated, summarize_order, is_same_string, format_money, PRODUCT_FIELDS, ORDER_FIELDS


# ─────────────────────────────────────────────
# Private Helpers
# ─────────────────────────────────────────────

def _fetch_orders_gql(query_filter: str) -> list:
    """Fetch all orders matching a Shopify query filter string, paginated."""
    query = f"""
    query ($cursor: String, $query: String) {{
        orders(first: 250, after: $cursor, query: $query) {{
            pageInfo {{ hasNextPage endCursor }}
            edges {{ node {{ {ORDER_FIELDS} }} }}
        }}
    }}
    """
    return gql_paginated(query, variables={"query": query_filter}, data_path=["orders"])


def _fetch_products_gql(query_filter: str = "status:active") -> list:
    """Fetch all products matching a query filter, paginated."""
    query = f"""
    query ($cursor: String, $query: String) {{
        products(first: 250, after: $cursor, query: $query) {{
            pageInfo {{ hasNextPage endCursor }}
            edges {{ node {{ {PRODUCT_FIELDS} }} }}
        }}
    }}
    """
    return gql_paginated(query, variables={"query": query_filter}, data_path=["products"])


def _order_revenue(order: dict) -> float:
    return float(order.get("totalPriceSet", {}).get("shopMoney", {}).get("amount", 0) or 0)


def _paid_orders_filter(start: date, end: date) -> str:
    return f'(financial_status:PAID OR financial_status:PENDING) AND created_at:>"{start}" AND created_at:<"{end}"'


# ─────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────

@tool(description="Returns today's date in ISO format (YYYY-MM-DD).")
def fetch_today_date() -> str:
    return f"Today's date is {date.today().isoformat()}"


@tool
def get_revenue_summary(
    iso_start_date: date,
    iso_end_date: date,
    top_n: int = 0,
    tag: str = "",
    product_name: str = "",
) -> dict:
    """
    Total revenue, order count, AOV, and top-selling products for a time period.

    Args:
        iso_start_date: Start date inclusive (YYYY-MM-DD).
        iso_end_date:   End date inclusive (YYYY-MM-DD).
        top_n:          Number of top products to return by units sold (0 = default 3).
        tag:            Optional product tag to restrict revenue to (e.g. "Wallets").
        product_name:   Optional approximate product name; fuzzy-matched against active titles.

    Returns:
        Dict: period_days, start_date, end_date, tag_filter, product_filter,
              total_revenue (PKR), total_orders, average_order_value (PKR),
              top_products [{product_title, total_units_sold}].
    """
    try:
        n = top_n if top_n > 0 else 3

        allowed_titles: set | None = None
        if tag or product_name:
            query_parts = ["status:active"]
            if tag:
                query_parts.append(f'tag:"{tag}"')
            filtered_products = _fetch_products_gql(" AND ".join(query_parts))
            all_titles = [p.get("title", "") for p in filtered_products if p.get("title")]

            if product_name:
                exact = is_same_string(product_name, all_titles)
                if exact:
                    allowed_titles = set(exact)
                else:
                    fuzzy_matches = process.extract(
                        product_name, all_titles,
                        scorer=fuzz.WRatio, processor=str.lower,
                        score_cutoff=65, limit=2,
                    )
                    matched = {m[0] for m in fuzzy_matches}
                    if not matched:
                        return {"error": "No matching product found. Please refine."}
                    if len(matched) > 1:
                        options = ", ".join(f'"{m}"' for m in matched)
                        return {"error": f"Multiple similar products found: [{options}]. Which one did you mean?"}
                    allowed_titles = matched
            else:
                allowed_titles = set(all_titles)

        orders = _fetch_orders_gql(_paid_orders_filter(iso_start_date, iso_end_date))

        total_revenue = 0.0
        total_orders = 0
        stats: dict = {}

        for order in orders:
            order_contributes = False
            order_revenue = _order_revenue(order)

            for edge in order.get("lineItems", {}).get("edges", []):
                item = edge["node"]
                title = item.get("title", "Unknown")

                if allowed_titles is not None and title not in allowed_titles:
                    continue

                qty = item.get("quantity", 0) or 0
                order_contributes = True
                stats.setdefault(title, {"total_units_sold": 0})
                stats[title]["total_units_sold"] += qty

            if order_contributes:
                total_revenue += order_revenue
                total_orders += 1

        ranked = sorted(
            [{"product_title": k, **v} for k, v in stats.items()],
            key=lambda x: x["total_units_sold"],
            reverse=True,
        )

        return {
            "period_days": (iso_end_date - iso_start_date).days + 1,
            "start_date": iso_start_date.isoformat()[:10],
            "end_date": iso_end_date.isoformat()[:10],
            "tag_filter": tag or None,
            "product_filter": next(iter(allowed_titles)) if allowed_titles and product_name else None,
            "total_revenue": format_money(total_revenue),
            "total_orders": total_orders,
            "average_order_value": format_money(total_revenue / total_orders if total_orders else 0),
            "top_products": ranked[:n],
        }
    except Exception as e:
        return {"error": f"Failed to get revenue summary: {e}"}


@tool
def get_unfulfilled_orders() -> dict:
    """
    All currently unfulfilled open orders — count, total value, and order list.

    Returns:
        Dict: count, total_value (PKR), orders (up to 20 summarized orders).
    """
    try:
        orders = _fetch_orders_gql("fulfillment_status:unfulfilled AND status:open")
        return {
            "count": len(orders),
            "total_value": format_money(sum(_order_revenue(o) for o in orders)),
            "orders": [summarize_order(o) for o in orders[:20]],
        }
    except Exception as e:
        return {"error": f"Failed to get unfulfilled orders: {e}"}


@tool
def get_low_inventory_products(threshold: int = 3) -> list:
    """
    Active product variants with inventory at or below a threshold, sorted most critical first.

    Args:
        threshold: Max inventory level to flag (default 3).

    Returns:
        List of {product_title, variant_title, inventory_quantity, sku}.
    """
    try:
        products = _fetch_products_gql("status:active")
        low_stock = []
        for p in products:
            for edge in p.get("variants", {}).get("edges", []):
                v = edge["node"]
                qty = v.get("inventoryQuantity", 0) or 0
                if qty <= threshold:
                    low_stock.append({
                        "product_title": p.get("title", ""),
                        "variant_title": v.get("title", "Default"),
                        "inventory_quantity": qty,
                        "sku": v.get("sku", "N/A"),
                    })
        return sorted(low_stock, key=lambda x: x["inventory_quantity"])
    except Exception as e:
        return [{"error": f"Failed to get low inventory products: {e}"}]


@tool
def compare_sales_periods(
    iso_start_date_period_1: date,
    iso_end_date_period_1: date,
    iso_start_date_period_2: date,
    iso_end_date_period_2: date,
) -> dict:
    """
    Side-by-side revenue and order count comparison between two date periods.

    Args:
        iso_start_date_period_1 / iso_end_date_period_1: Current (more recent) period.
        iso_start_date_period_2 / iso_end_date_period_2: Previous period to compare against.

    Returns:
        Dict: current_period stats, previous_period stats, and changes (absolute + %).
    """
    try:
        def fetch_stats(start, end):
            orders = _fetch_orders_gql(_paid_orders_filter(start, end))
            return {"order_count": len(orders), "revenue": sum(_order_revenue(o) for o in orders)}

        curr = fetch_stats(iso_start_date_period_1, iso_end_date_period_1)
        prev = fetch_stats(iso_start_date_period_2, iso_end_date_period_2)
        rev_change = curr["revenue"] - prev["revenue"]
        ord_change = curr["order_count"] - prev["order_count"]

        return {
            "current_period": {
                "start": iso_start_date_period_1.isoformat()[:10],
                "end": iso_end_date_period_1.isoformat()[:10],
                "revenue": format_money(curr["revenue"]),
                "order_count": curr["order_count"],
            },
            "previous_period": {
                "start": iso_start_date_period_2.isoformat()[:10],
                "end": iso_end_date_period_2.isoformat()[:10],
                "revenue": format_money(prev["revenue"]),
                "order_count": prev["order_count"],
            },
            "changes": {
                "revenue_change": format_money(rev_change),
                "revenue_change_pct": f"{(rev_change / prev['revenue'] * 100) if prev['revenue'] else 0:+.1f}%",
                "order_change": ord_change,
                "order_change_pct": f"{(ord_change / prev['order_count'] * 100) if prev['order_count'] else 0:+.1f}%",
            },
        }
    except Exception as e:
        return {"error": f"Failed to compare sales periods: {e}"}


@tool
def get_refunded_orders(iso_start_date: date, iso_end_date: date) -> list:
    """
    Fully and partially refunded orders in a date range.

    Args:
        iso_start_date: Start date inclusive (YYYY-MM-DD).
        iso_end_date:   End date inclusive (YYYY-MM-DD).

    Returns:
        List of summarized order dicts with refund transaction details.
    """
    try:
        refunded = _fetch_orders_gql(
            f'financial_status:refunded AND created_at:>"{iso_start_date}" AND created_at:<"{iso_end_date}"'
        )
        partial = _fetch_orders_gql(
            f'financial_status:partially_refunded AND created_at:>"{iso_start_date}" AND created_at:<"{iso_end_date}"'
        )
        return [summarize_order(o) for o in refunded + partial]
    except Exception as e:
        return [{"error": f"Failed to get refunded orders: {e}"}]


@tool
def get_zero_sales_products(iso_start_date: date, iso_end_date: date) -> list:
    """
    Active products with zero paid sales in a period — identifies potential dead stock.

    Args:
        iso_start_date: Start date inclusive (YYYY-MM-DD).
        iso_end_date:   End date inclusive (YYYY-MM-DD).

    Returns:
        Sorted list of product title strings with no sales, or a confirmation message if all sold.
    """
    try:
        all_titles = {p.get("title") for p in _fetch_products_gql("status:active")}
        orders = _fetch_orders_gql(_paid_orders_filter(iso_start_date, iso_end_date))
        sold = {
            edge["node"].get("title")
            for o in orders
            for edge in o.get("lineItems", {}).get("edges", [])
        }
        zero = sorted(all_titles - sold)
        return zero or ["All products have had at least one sale in this period."]
    except Exception as e:
        return [f"Error: Failed to get zero-sales products: {e}"]


@tool
def get_recent_orders(iso_start_date: date, iso_end_date: date) -> list:
    """
    Orders placed in a given date range with customer details and order values.

    Args:
        iso_start_date: Start date inclusive (YYYY-MM-DD).
        iso_end_date:   End date inclusive (YYYY-MM-DD).

    Returns:
        List of summarized order dicts.
    """
    try:
        orders = _fetch_orders_gql(
            f'created_at:>"{iso_start_date}" AND created_at:<"{iso_end_date}"'
        )
        return [summarize_order(o) for o in orders]
    except Exception as e:
        return [{"error": f"Failed to get recent orders: {e}"}]


# ─────────────────────────────────────────────
# Exported tool list
# ─────────────────────────────────────────────

ADMIN_TOOLS = [
    fetch_today_date,
    get_revenue_summary,
    get_unfulfilled_orders,
    get_low_inventory_products,
    compare_sales_periods,
    get_refunded_orders,
    get_zero_sales_products,
    get_recent_orders,
]