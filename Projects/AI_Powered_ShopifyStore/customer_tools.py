"""
customer_tools.py — Tools available to the Customer Support Agent.

All Shopify data is fetched via the Admin GraphQL API.

Tools:
1. search_products   — Browse/filter products by tags, price, color, name, stock.
2. get_best_sellers  — Featured collection / best-selling products.
3. get_order_status  — Order tracking by order number.
4. get_store_policies — Return, refund, and discount policy (static).
"""

from langchain_core.tools import tool
from utils import (
    gql,
    gql_paginated,
    filter_products_by_price,
    filter_products_by_name,
    summarize_product,
    summarize_order,
    PRODUCT_FIELDS,
    ORDER_FIELDS,
)


# ─────────────────────────────────────────────────────────────
# Tool 1: Search / Browse Products
# ─────────────────────────────────────────────────────────────

@tool
def search_products(
    tags: list[str] = [],
    max_price: float = 0.0,
    color: str = "",
    product_name: str = "",
    in_stock_only: bool = False,
) -> list:
    """
    Search Silk Skin products by one or more tags, price, color, product name, and stock status.

    ── WHEN TO USE ──────────────────────────────────────────────────────────────────
    Use for any product browsing, filtering, or searching request from a customer.

    ── TAG SELECTION GUIDE ──────────────────────────────────────────────────────────
    Pass ALL relevant tags in one call (OR logic — products matching ANY tag included).
    Do NOT call this tool multiple times for multi-tag searches.

    VALID TAGS (exact spelling required):
      "Wallet"              → Men's/general leather wallets
      "Ladies Wallet"       → Wallets for women
      "Card Holder"         → Slim minimal card holders
      "Handbags"            → Women's handbags
      "Bags"                → General non-travel bags
      "Travel"              → Travel bags, passport holders, travel accessories
      "Gifts"               → Luxury gift sets
      "Accessories"         → Leather accessories
      "featured collection" → Best-sellers / featured items

    ── PRODUCT NAME SEARCH ──────────────────────────────────────────────────────────
    Pass approximate product name in 'product_name'. Fuzzy matching is applied —
    exact title not required. Use name without 'Product' keyword.

    ── PRICE ────────────────────────────────────────────────────────────────────────
    All prices are in Pakistani Rupees (PKR). max_price=0.0 means no price filter.

    ── COLOR ────────────────────────────────────────────────────────────────────────
    Pass a plain color word: "black", "brown", "red", "tan", etc.
    If no color match found, falls back to showing all results so the agent can
    inform the customer which colors ARE available.

    Args:
        tags: List of tags to filter by (OR logic). Empty = all products.
        max_price: Maximum price in PKR. 0.0 = no filter.
        color: Color keyword to search in variant titles and product description.
        product_name: Approximate name of product for fuzzy title matching.
        in_stock_only: If True, only return products with inventory > 0.

    Returns:
        List of product summaries: {id, title, tags, price_range, in_stock, variants}.
    """
    try:
        query = f"""
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

        all_products = []
        seen_ids = set()

        if tags:
            # Fetch each tag separately and deduplicate by product ID
            for tag in tags:
                tag_products = gql_paginated(
                    query,
                    variables={"query": f'tag:"{tag}" AND status:active'},
                    data_path=["products"],
                )
                for p in tag_products:
                    pid = p.get("id")
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        all_products.append(p)
        else:
            # No tag filter — fetch all active products
            all_products = gql_paginated(
                query,
                variables={"query": "status:active"},
                data_path=["products"],
            )

        # Summarize first (converts GraphQL shape to flat dicts)
        summarized = [summarize_product(p) for p in all_products]

        # Filter by product name (fuzzy)
        if product_name:
            print("1")
            summarized = filter_products_by_name(summarized, product_name)
            if isinstance(summarized, str):
                return summarized                

        # Filter by max price
        if max_price and max_price > 0:
            summarized = filter_products_by_price(summarized, max_price)

        # Filter by color
        if color:
            color_lower = color.lower()
            color_filtered = []
            for p in summarized:
                # Check variant titles
                variant_match = any(
                    color_lower in v.get("title", "").lower()
                    for v in p.get("variants", [])
                )
                # Check product title and description
                title_match = color_lower in p.get("title", "").lower()
                desc_match = color_lower in p.get("description", "").lower()

                if variant_match or title_match or desc_match:
                    color_filtered.append(p)

            # If color yields results use them; else return all so agent can suggest available colors
            summarized = color_filtered if color_filtered else summarized

        # Filter by stock
        if in_stock_only:
            summarized = [
                p for p in summarized
                if any(v.get("inventory_quantity", 0) > 0 for v in p.get("variants", []))
            ]

        return summarized

    except Exception as e:
        return f"Failed to search products: {str(e)}"


# ─────────────────────────────────────────────────────────────
# Tool 2: Get Best-Selling Products
# ─────────────────────────────────────────────────────────────

@tool
def get_best_sellers(limit: int = 5) -> list:
    """
    Retrieve products from the featured collection — Silk Skin's best-sellers.

    Use this when a customer asks:
    - "What are your best-sellers?"
    - "What's popular right now?"
    - "Show me your top products."
    - "What do most people buy?"

    Args:
        limit: Number of products to return (default 5, max 10).

    Returns:
        List of product summaries from the featured collection.
    """
    try:
        limit = min(limit, 10)
        query = f"""
        query ($cursor: String) {{
            products(first: 250, after: $cursor, query: "tag:\\"featured collection\\" AND status:active") {{
                pageInfo {{ hasNextPage endCursor }}
                edges {{
                    node {{
                        {PRODUCT_FIELDS}
                    }}
                }}
            }}
        }}
        """
        products = gql_paginated(query, variables={}, data_path=["products"])
        return [summarize_product(p) for p in products[:limit]]

    except Exception as e:
        return [{"error": f"Failed to get best sellers: {str(e)}"}]


# ─────────────────────────────────────────────────────────────
# Tool 3: Get Order Status
# ─────────────────────────────────────────────────────────────

@tool
def get_order_status(order_number: str) -> dict:
    """
    Look up a customer's order status, fulfillment info, and tracking details.

    Use this when a customer provides an order number and asks:
    - "Where is my order?"
    - "Has my order shipped?"
    - "Track my order #45821"
    - "When will Order #45821 arrive?"

    The order number may be formatted as '#45821' or just '45821'.

    Args:
        order_number: The order number string, e.g. '45821' or '#45821'.

    Returns:
        Order summary dict with status, fulfillment, tracking, and line items.
        Returns an error dict if the order is not found.
    """
    try:
        clean_number = order_number.lstrip("#").strip()

        query = f"""
        query ($query: String!) {{
            orders(first: 1, query: $query) {{
                edges {{
                    node {{
                        {ORDER_FIELDS}
                    }}
                }}
            }}
        }}
        """

        # Shopify order name format is '#45821'
        data = gql(query, {"query": f'name:"#{clean_number}"'})
        edges = data.get("orders", {}).get("edges", [])

        if not edges:
            # Retry without '#' prefix in case store uses plain numbering
            data = gql(query, {"query": f'name:"{clean_number}"'})
            edges = data.get("orders", {}).get("edges", [])

        if not edges:
            return {
                "error": (
                    f"No order found with number #{clean_number}. "
                    "Please double-check the order number and try again."
                )
            }

        return summarize_order(edges[0]["node"])

    except Exception as e:
        return {"error": f"Failed to retrieve order status: {str(e)}"}


# ─────────────────────────────────────────────────────────────
# Tool 4: Get Store Policies
# ─────────────────────────────────────────────────────────────

@tool
def get_store_policies(which_policy: str) -> dict:
    """
    Return Silk Skin's return, refund, and discount policies.

    Use this when a customer asks about:
    - Return policy ("Can I return this?", "What's your return policy?")
    - Refund process ("How do I get a refund?", "I received a damaged item")
    - Discounts or promo codes ("Do you have discount codes?")
    - Exchange policy
    
    Args:
        which_policy: Specify which policy info to return. Options:
            'return_policy', 'refund_policy', 'damaged_item_process', or 'discounts'
    """
    return {
        "return_policy": (
            "Silk Skin offers a 14-day return window from the date of delivery. "
            "Items must be unused, in original condition, and returned in original packaging. "
            "To initiate a return, contact our support team with your order number. "
            "Custom or personalized items are not eligible for return."
        ),
        "refund_policy": (
            "Once your returned item is received and inspected, we will notify you of the approval. "
            "Approved refunds are processed within 5-7 business days to the original payment method."
        ),
        "damaged_item_process": (
            "If you received a damaged item: "
            "1. Take clear photos of the damage. "
            "2. Contact support with your order number and photos. "
            "3. We will process a replacement or full refund within 48 hours of verification — "
            "no return required."
        ),
        "discounts": (
            "Silk Skin occasionally offers promotional discounts to newsletter subscribers "
            "and loyal customers. No universal promo code is currently active. "
            "Subscribe to our newsletter or follow our social media for exclusive offers."
        ),
    }


# ─────────────────────────────────────────────────────────────
# Exported tool list
# ─────────────────────────────────────────────────────────────

CUSTOMER_TOOLS = [
    search_products,
    get_best_sellers,
    get_order_status,
    get_store_policies,
]