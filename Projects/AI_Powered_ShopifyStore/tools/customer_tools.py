"""
tools/customer_tools.py

Customer-facing tools for the customer_support_agent.

4 tools that collectively handle all customer use-cases:
  1. search_products          — keyword, price filter, size/color availability, similar products
  2. get_best_selling_products — top sellers by units sold over a time window
  3. track_order              — order status, fulfillment, and shipping tracking
  4. get_store_policy         — shipping, returns, discounts, damaged item guidance
"""

from typing import Literal, Optional
from langchain_core.tools import tool

from utils import (
    ShopifyClient,
    format_product_summary,
    extract_product_sales,
    iso_days_ago,
    get_logger,
)

logger = get_logger("customer_tools")
_client = ShopifyClient()


# ─── 1. Product Search ────────────────────────────────────────────────────────

@tool
def search_products(
    keyword: str = "",
    max_price: Optional[float] = None,
    size: str = "",
    color: str = "",
    find_similar_to: str = "",
    limit: int = 10,
) -> str:
    """
    Unified product search. Handles keyword search, price filtering, variant
    availability (size/color), and finding similar products — all in one call.

    Use this tool for any of these requests:
      - "Find summer dresses under $50"     → keyword="summer dress", max_price=50
      - "Do you have this in size medium?"  → keyword="<product>", size="medium"
      - "Is this available in black?"       → keyword="<product>", color="black"
      - "Show me products similar to X"     → find_similar_to="X"
      - "Show me all sneakers"              → keyword="sneakers"

    Args:
        keyword: Product name or search term.
        max_price: Upper price limit in USD. Filters to variants at or below this price.
        size: Variant size to check, e.g. 'medium', 'M', 'XL'.
        color: Variant color to check, e.g. 'black', 'red'.
        find_similar_to: Product title to base a similarity search on (tags + product type).
        limit: Max results to return (default 10).
    """
    try:
        # ── Similarity Mode ───────────────────────────────────────────────────
        if find_similar_to:
            ref_hits = _client.search_products(query=find_similar_to, limit=3)
            if not ref_hits:
                return f"Could not find a product matching '{find_similar_to}' to base recommendations on."

            ref = ref_hits[0]
            ref_tags = {t.strip().lower() for t in ref.get("tags", "").split(",") if t.strip()}
            ref_type = ref.get("product_type", "").lower()

            catalog = _client.get_all_products(limit=150)
            scored = []
            for p in catalog:
                if p["id"] == ref["id"]:
                    continue
                p_tags = {t.strip().lower() for t in p.get("tags", "").split(",") if t.strip()}
                shared = len(ref_tags & p_tags)
                type_match = int(bool(ref_type and p.get("product_type", "").lower() == ref_type))
                score = shared * 2 + type_match
                if score > 0:
                    scored.append((score, p))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = [p for _, p in scored[:limit]]

            if not top:
                return f"No similar products found for '{ref.get('title')}'."

            lines = [f"Products similar to '{ref.get('title')}':"]
            for p in top:
                lines.append(format_product_summary(p))
            return "\n".join(lines)

        # ── Standard Search Mode ──────────────────────────────────────────────
        fetch_limit = 250 if (max_price is not None or size or color) else limit
        products = _client.search_products(query=keyword, limit=fetch_limit)

        if not products:
            suffix = f" matching '{keyword}'" if keyword else ""
            return f"No products found{suffix}."

        # Price filter
        if max_price is not None:
            products = [
                p for p in products
                if any(float(v.get("price") or 0) <= max_price for v in p.get("variants", []))
            ]
            if not products:
                suffix = f" matching '{keyword}'" if keyword else ""
                return f"No products found under ${max_price:.2f}{suffix}."

        # Variant filter (size / color)
        if size or color:
            results = []
            for p in products:
                matched = []
                for v in p.get("variants", []):
                    v_title = v.get("title", "").lower()
                    if (not size or size.lower() in v_title) and (not color or color.lower() in v_title):
                        inv = v.get("inventory_quantity")
                        stock = (
                            "In Stock" if isinstance(inv, int) and inv > 0
                            else "Out of Stock" if isinstance(inv, int)
                            else "Availability Unknown"
                        )
                        matched.append(f"    └ {v.get('title')} | ${v.get('price')} | {stock}")
                if matched:
                    results.append(f"• {p.get('title')}")
                    results.extend(matched)

            if not results:
                filter_desc = " / ".join(filter(None, [size, color]))
                suffix = f" for '{keyword}'" if keyword else ""
                return f"No variants matching '{filter_desc}' found{suffix}."

            parts = [keyword, f"size: {size}" if size else "", f"color: {color}" if color else ""]
            header = ", ".join(p for p in parts if p)
            return f"Variant Availability ({header}):\n" + "\n".join(results)

        # Plain results
        display = products[:limit]
        suffix = f" for '{keyword}'" if keyword else ""
        lines = [f"Found {len(display)} product(s){suffix}:"]
        for p in display:
            lines.append(format_product_summary(p))
        return "\n".join(lines)

    except Exception as e:
        logger.error("search_products error: %s", e)
        return f"Error searching products: {e}"


# ─── 2. Best Sellers ──────────────────────────────────────────────────────────

@tool
def get_best_selling_products(days: int = 30, top_n: int = 5) -> str:
    """
    Return the top-selling products by units sold over the past N days.
    Use for: "best sellers", "popular items", "what's trending".

    Args:
        days: Lookback window in days (default 30).
        top_n: Number of top products to return (default 5).
    """
    try:
        orders = _client.get_orders_in_date_range(
            created_at_min=iso_days_ago(days),
            created_at_max=iso_days_ago(0),
        )
        if not orders:
            return f"No sales data available for the last {days} days."

        sales = extract_product_sales(orders)
        if not sales:
            return "No product sales data found in recent orders."

        ranked = sorted(sales.items(), key=lambda x: x[1]["quantity"], reverse=True)[:top_n]

        lines = [f"Top {top_n} Best-Selling Products (last {days} days):"]
        for i, (title, data) in enumerate(ranked, 1):
            lines.append(f"  {i}. {title} — {data['quantity']} sold")
        return "\n".join(lines)

    except Exception as e:
        logger.error("get_best_selling_products error: %s", e)
        return f"Error fetching best-selling products: {e}"


# ─── 3. Order Tracking ────────────────────────────────────────────────────────

@tool
def track_order(order_id: int) -> str:
    """
    Look up an order by its numeric ID and return full status, fulfillment,
    line items, and shipping tracking details.

    Use for: "Where is my order?", "What's the status of #XXXXX?"

    Args:
        order_id: The numeric Shopify order ID (e.g. 45821).
    """
    try:
        order = _client.get_order(order_id)
        if not order:
            return (
                f"Order #{order_id} not found. Please double-check the number from "
                "your confirmation email."
            )

        lines = [
            f"Order #{order_id}",
            f"  Placed:       {order.get('created_at', '')[:10]}",
            f"  Payment:      {order.get('financial_status', 'unknown')}",
            f"  Fulfillment:  {order.get('fulfillment_status') or 'unfulfilled'}",
            f"  Total:        ${order.get('total_price', '0')}",
        ]

        items = order.get("line_items", [])
        if items:
            lines.append("  Items:")
            for i in items[:5]:
                lines.append(f"    - {i.get('name')} x{i.get('quantity')}")

        fulfillments = order.get("fulfillments", [])
        if fulfillments:
            lines.append("  Shipping:")
            for f in fulfillments:
                carrier = f.get("tracking_company") or "N/A"
                number = f.get("tracking_number") or "N/A"
                url = f.get("tracking_url") or ""
                lines.append(f"    Carrier: {carrier} | Tracking #: {number} | Status: {f.get('status', 'N/A')}")
                if url:
                    lines.append(f"    Track: {url}")
        else:
            lines.append("  Shipping: Not yet shipped. You'll receive a notification when it dispatches.")

        return "\n".join(lines)

    except Exception as e:
        logger.error("track_order error: %s", e)
        return (
            f"Unable to retrieve order #{order_id}. "
            f"Please try again or contact our support team. (Error: {e})"
        )


# ─── 4. Store Policies & Support ─────────────────────────────────────────────

@tool
def get_store_policy(
    topic: Literal["shipping", "returns", "discounts", "damaged_item"],
    destination: str = "",
    order_id: Optional[int] = None,
) -> str:
    """
    Return store policy information or guide customers through common support issues.

    Use this for:
      - "How long does shipping take to California?"  → topic="shipping", destination="California"
      - "What is your return policy?"                 → topic="returns"
      - "Do you have any promo codes?"                → topic="discounts"
      - "I received a damaged item"                   → topic="damaged_item", order_id=<id>

    Args:
        topic: One of 'shipping', 'returns', 'discounts', 'damaged_item'.
        destination: Used with topic='shipping' to tailor delivery estimates.
        order_id: Used with topic='damaged_item' to verify and reference the order.
    """
    try:
        if topic == "shipping":
            response = (
                "Shipping Options:\n"
                "  • Standard  — 5–7 business days | Free on orders over $50\n"
                "  • Express   — 2–3 business days | $12.99\n"
                "  • Overnight — 1 business day    | $24.99\n\n"
                "  Orders are processed within 1–2 business days after payment confirmation."
            )
            if destination:
                response += (
                    f"\n\n  Delivery to {destination}: approx. 5–7 days (Standard) "
                    f"or 2–3 days (Express) from dispatch."
                )
            return response

        elif topic == "returns":
            return (
                "Return & Refund Policy:\n"
                "  • 30-day return window from delivery date.\n"
                "  • Items must be unused, unwashed, and in original packaging.\n"
                "  • To initiate: email support@yourstore.com with your order number.\n"
                "  • Refunds issued within 5–7 business days of receiving your return.\n"
                "  • Final sale and personalized items cannot be returned.\n"
                "  • Damaged or defective items: full refund or replacement with no return required."
            )

        elif topic == "discounts":
            # Attempt live fetch first
            try:
                data = _client._get("price_rules.json", params={"limit": 20})
                active = [r for r in data.get("price_rules", []) if r.get("status") == "enabled"]
                if active:
                    lines = ["Current Promotions:"]
                    for r in active:
                        val = float(r.get("value", 0))
                        vtype = r.get("value_type", "")
                        amount = f"{abs(val):.0f}% off" if vtype == "percentage" else f"${abs(val):.2f} off"
                        entry = f"  • {r.get('title', 'Promo')}: {amount}"
                        if r.get("ends_at"):
                            entry += f" (expires {r['ends_at'][:10]})"
                        lines.append(entry)
                    return "\n".join(lines)
            except Exception as promo_err:
                logger.warning("Live price rules unavailable: %s", promo_err)

            return (
                "Current Offers:\n"
                "  • WELCOME10 — 10% off your first order.\n"
                "  • Free standard shipping on orders over $50.\n"
                "  • Newsletter subscribers get early access to sales and exclusive codes."
            )

        elif topic == "damaged_item":
            order_context = ""
            if order_id:
                try:
                    order = _client.get_order(order_id)
                    if order:
                        items = order.get("line_items", [])
                        names = ", ".join(i.get("name", "") for i in items[:3])
                        order_context = f"\n  We can see your order contained: {names}."
                    else:
                        order_context = f"\n  (Order #{order_id} could not be verified — please include it in your email.)"
                except Exception:
                    pass

            ref = f"order #{order_id}" if order_id else "your order"
            return (
                f"We're sorry about the damaged item!{order_context}\n\n"
                "Here's how we'll make it right:\n"
                f"  1. Take a clear photo of the damaged item and packaging.\n"
                f"  2. Email support@yourstore.com with:\n"
                f"       - Order number ({ref})\n"
                f"       - Brief description of the damage\n"
                f"       - Photos attached\n"
                f"  3. We'll reply within 24 hours with a full replacement or refund — your choice.\n\n"
                "  You do NOT need to return the damaged item."
            )

        return f"Unknown topic '{topic}'. Valid options: shipping, returns, discounts, damaged_item."

    except Exception as e:
        logger.error("get_store_policy error (topic=%s): %s", topic, e)
        return f"Error retrieving policy for '{topic}': {e}"


# ─── Tool Registry ────────────────────────────────────────────────────────────

CUSTOMER_TOOLS = [
    search_products,
    get_best_selling_products,
    track_order,
    get_store_policy,
]