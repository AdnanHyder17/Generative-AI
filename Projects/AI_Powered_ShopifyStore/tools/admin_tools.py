"""
tools/admin_tools.py

Admin-only tools for the admin_support_agent.

5 tools that collectively cover all admin analytics use-cases:
  1. get_sales_report        — today / last N days / month-over-month + AOV
  2. get_product_performance — top sellers, unsold products
  3. get_inventory_status    — low-stock alerts across all variants
  4. get_orders_report       — unfulfilled orders, refunded orders
  5. get_customer_insights   — top repeat customers by order count / spend
"""

from collections import defaultdict
from typing import Literal
from langchain_core.tools import tool

from utils import (
    ShopifyClient,
    calculate_total_sales,
    calculate_average_order_value,
    extract_product_sales,
    format_order_summary,
    format_price,
    iso_days_ago,
    start_of_today_iso,
    start_of_month_iso,
    start_of_last_month_iso,
    end_of_last_month_iso,
    get_logger,
)

logger = get_logger("admin_tools")
_client = ShopifyClient()


# ─── 1. Sales Report ──────────────────────────────────────────────────────────

@tool
def get_sales_report(
    period: Literal["today", "last_7_days", "last_30_days", "this_month", "month_over_month"] = "today",
) -> str:
    """
    Comprehensive sales report for a given time period. Returns revenue, order count,
    average order value, and daily breakdown where applicable.

    Use this for:
      - "Show me today's sales"                           → period="today"
      - "Sales for the last 7 days"                       → period="last_7_days"
      - "Show me sales performance for the last 30 days"  → period="last_30_days"
      - "What's the average order value this month?"      → period="this_month"
      - "Compare this month vs last month"                → period="month_over_month"

    Args:
        period: One of 'today', 'last_7_days', 'last_30_days', 'this_month', 'month_over_month'.
    """
    try:
        if period == "today":
            orders = _client.get_orders_in_date_range(
                created_at_min=start_of_today_iso(),
                created_at_max=iso_days_ago(0),
            )
            total = calculate_total_sales(orders)
            aov = total / len(orders) if orders else 0
            return (
                f"Today's Sales:\n"
                f"  Orders:  {len(orders)}\n"
                f"  Revenue: {format_price(total)}\n"
                f"  AOV:     {format_price(aov)}"
            )

        elif period in ("last_7_days", "last_30_days"):
            days = 7 if period == "last_7_days" else 30
            orders = _client.get_orders_in_date_range(
                created_at_min=iso_days_ago(days),
                created_at_max=iso_days_ago(0),
            )
            if not orders:
                return f"No orders in the last {days} days."

            by_date: dict = defaultdict(lambda: {"count": 0, "revenue": 0.0})
            for o in orders:
                date = o.get("created_at", "")[:10]
                by_date[date]["count"] += 1
                try:
                    by_date[date]["revenue"] += float(o.get("total_price") or 0)
                except (TypeError, ValueError):
                    pass

            total = calculate_total_sales(orders)
            aov = calculate_average_order_value(orders)

            lines = [f"Sales Performance — Last {days} Days:"]
            for date in sorted(by_date):
                d = by_date[date]
                lines.append(f"  {date}: {d['count']} order(s) | {format_price(d['revenue'])}")
            lines.append(f"\n  Total:   {len(orders)} orders | {format_price(total)}")
            lines.append(f"  AOV:     {format_price(aov)}")
            return "\n".join(lines)

        elif period == "this_month":
            orders = _client.get_orders_in_date_range(
                created_at_min=start_of_month_iso(),
                created_at_max=iso_days_ago(0),
            )
            if not orders:
                return "No orders this month yet."
            total = calculate_total_sales(orders)
            aov = calculate_average_order_value(orders)
            return (
                f"This Month's Sales:\n"
                f"  Orders:  {len(orders)}\n"
                f"  Revenue: {format_price(total)}\n"
                f"  AOV:     {format_price(aov)}"
            )

        elif period == "month_over_month":
            this_orders = _client.get_orders_in_date_range(
                created_at_min=start_of_month_iso(),
                created_at_max=iso_days_ago(0),
            )
            last_orders = _client.get_orders_in_date_range(
                created_at_min=start_of_last_month_iso(),
                created_at_max=end_of_last_month_iso(),
            )
            this_rev = calculate_total_sales(this_orders)
            last_rev = calculate_total_sales(last_orders)
            rev_pct = ((this_rev - last_rev) / last_rev * 100) if last_rev else float("inf")
            order_diff = len(this_orders) - len(last_orders)

            return (
                f"Month-over-Month Comparison:\n"
                f"  This Month: {len(this_orders):>4} orders | {format_price(this_rev)}\n"
                f"  Last Month: {len(last_orders):>4} orders | {format_price(last_rev)}\n"
                f"  Change:     {order_diff:+d} orders | Revenue {rev_pct:+.1f}%"
            )

        return f"Unknown period '{period}'."

    except Exception as e:
        logger.error("get_sales_report error (period=%s): %s", period, e)
        return f"Error generating sales report: {e}"


# ─── 2. Product Performance ───────────────────────────────────────────────────

@tool
def get_product_performance(
    report_type: Literal["top_sellers", "unsold"] = "top_sellers",
    days: int = 30,
    top_n: int = 5,
) -> str:
    """
    Analyse product-level sales performance.

    Use this for:
      - "Top 5 selling products this month"           → report_type="top_sellers", days=30
      - "Which products haven't sold in 30 days?"     → report_type="unsold", days=30

    Args:
        report_type: 'top_sellers' for best performers, 'unsold' for stale active products.
        days: Lookback window in days (default 30). For top_sellers this scopes the data;
              for unsold this defines the "no sale" threshold.
        top_n: Number of top products to return (only used with report_type='top_sellers').
    """
    try:
        orders = _client.get_orders_in_date_range(
            created_at_min=iso_days_ago(days),
            created_at_max=iso_days_ago(0),
        )

        if report_type == "top_sellers":
            if not orders:
                return f"No orders in the last {days} days."

            sales = extract_product_sales(orders)
            ranked = sorted(sales.items(), key=lambda x: x[1]["quantity"], reverse=True)[:top_n]

            lines = [f"Top {top_n} Products (last {days} days):"]
            for i, (title, data) in enumerate(ranked, 1):
                lines.append(
                    f"  {i}. {title}\n"
                    f"     Sold: {data['quantity']} units | Revenue: {format_price(data['revenue'])}"
                )
            return "\n".join(lines)

        elif report_type == "unsold":
            sold_titles = set()
            for o in orders:
                for item in o.get("line_items", []):
                    sold_titles.add(item.get("title", "").strip().lower())

            all_products = _client.get_all_products(limit=250)
            unsold = [
                p for p in all_products
                if p.get("title", "").strip().lower() not in sold_titles
                and p.get("status") == "active"
            ]

            if not unsold:
                return f"All active products have had at least one sale in the last {days} days."

            lines = [f"Active Products With No Sales in Last {days} Days ({len(unsold)}):"]
            for p in unsold[:25]:
                lines.append(f"  • {p.get('title', 'Unknown')}")
            if len(unsold) > 25:
                lines.append(f"  ... and {len(unsold) - 25} more.")
            return "\n".join(lines)

        return f"Unknown report_type '{report_type}'. Use 'top_sellers' or 'unsold'."

    except Exception as e:
        logger.error("get_product_performance error (type=%s): %s", report_type, e)
        return f"Error generating product performance report: {e}"


# ─── 3. Inventory Status ──────────────────────────────────────────────────────

@tool
def get_inventory_status(threshold: int = 10) -> str:
    """
    List all product variants with inventory at or below the threshold.
    Use for: "Which products are low in inventory?", "Show me stock alerts."

    Args:
        threshold: Units-on-hand threshold. Variants at or below this are flagged (default 10).
    """
    try:
        products = _client.get_all_products(limit=250)
        out_of_stock = []
        low_stock = []

        for p in products:
            if p.get("status") != "active":
                continue
            for v in p.get("variants", []):
                inv = v.get("inventory_quantity")
                if not isinstance(inv, int):
                    continue
                variant_label = (
                    f"{p['title']}"
                    + (f" — {v.get('title')}" if v.get("title") != "Default Title" else "")
                )
                if inv == 0:
                    out_of_stock.append(f"  ⛔ {variant_label} | Stock: 0")
                elif inv <= threshold:
                    low_stock.append(f"  ⚠️  {variant_label} | Stock: {inv}")

        if not out_of_stock and not low_stock:
            return f"All active products are well-stocked (threshold: {threshold} units)."

        lines = [f"Inventory Alerts (threshold ≤ {threshold} units):"]
        if out_of_stock:
            lines.append(f"\n  Out of Stock ({len(out_of_stock)}):")
            lines.extend(out_of_stock)
        if low_stock:
            lines.append(f"\n  Low Stock ({len(low_stock)}):")
            lines.extend(low_stock)
        return "\n".join(lines)

    except Exception as e:
        logger.error("get_inventory_status error: %s", e)
        return f"Error checking inventory: {e}"


# ─── 4. Orders Report ─────────────────────────────────────────────────────────

@tool
def get_orders_report(
    filter_type: Literal["unfulfilled", "refunded"] = "unfulfilled",
    days: int = 7,
) -> str:
    """
    Return a filtered order report for admin review.

    Use this for:
      - "How many orders are unfulfilled?"          → filter_type="unfulfilled"
      - "List refunded orders from this week"       → filter_type="refunded", days=7

    Args:
        filter_type: 'unfulfilled' for open unshipped orders, 'refunded' for orders with refunds.
        days: Lookback window for refunded orders (default 7). Ignored for unfulfilled.
    """
    try:
        if filter_type == "unfulfilled":
            orders = _client.get_unfulfilled_orders()
            if not orders:
                return "No unfulfilled orders — all current orders are fulfilled or cancelled."

            lines = [f"Unfulfilled Orders: {len(orders)}"]
            for o in orders[:20]:
                lines.append(f"  {format_order_summary(o)}")
            if len(orders) > 20:
                lines.append(f"  ... and {len(orders) - 20} more.")
            return "\n".join(lines)

        elif filter_type == "refunded":
            orders = _client.get_refunded_orders(created_at_min=iso_days_ago(days))
            if not orders:
                return f"No refunded orders in the last {days} days."

            total_refunded = 0.0
            lines = [f"Refunded Orders — Last {days} Days ({len(orders)}):"]
            for o in orders:
                refund_amount = sum(
                    float(t.get("amount", 0))
                    for r in o.get("refunds", [])
                    for t in r.get("transactions", [])
                )
                total_refunded += refund_amount
                lines.append(
                    f"  {format_order_summary(o)} | Refunded: {format_price(refund_amount)}"
                )
            lines.append(f"\n  Total Refunded: {format_price(total_refunded)}")
            return "\n".join(lines)

        return f"Unknown filter_type '{filter_type}'. Use 'unfulfilled' or 'refunded'."

    except Exception as e:
        logger.error("get_orders_report error (filter=%s): %s", filter_type, e)
        return f"Error generating orders report: {e}"


# ─── 5. Customer Insights ─────────────────────────────────────────────────────

@tool
def get_customer_insights(top_n: int = 10) -> str:
    """
    Return the top repeat customers ranked by order count and total spend.
    Use for: "Who are my top customers?", "Show repeat buyers."

    Args:
        top_n: Number of top customers to display (default 10).
    """
    try:
        customers = _client.get_customers(limit=250)
        repeat = [c for c in customers if c.get("orders_count", 0) > 1]
        repeat.sort(key=lambda c: (c.get("orders_count", 0), float(c.get("total_spent") or 0)), reverse=True)
        top = repeat[:top_n]

        if not top:
            return "No repeat customers found yet."

        lines = [f"Top {top_n} Repeat Customers:"]
        for i, c in enumerate(top, 1):
            name = f"{c.get('first_name', '')} {c.get('last_name', '')}".strip() or "Unknown"
            email = c.get("email", "N/A")
            orders = c.get("orders_count", 0)
            spent = format_price(c.get("total_spent") or 0)
            lines.append(f"  {i:>2}. {name} ({email}) | Orders: {orders} | Total Spent: {spent}")
        return "\n".join(lines)

    except Exception as e:
        logger.error("get_customer_insights error: %s", e)
        return f"Error fetching customer insights: {e}"


# ─── Tool Registry ────────────────────────────────────────────────────────────

ADMIN_TOOLS = [
    get_sales_report,
    get_product_performance,
    get_inventory_status,
    get_orders_report,
    get_customer_insights,
]