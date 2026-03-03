"""
utils.py — Shared utility functions for Silk Skin AI Agent system.

All Shopify communication uses the Admin GraphQL API (2026-01).
GraphQL endpoint: https://{store}/admin/api/2026-01/graphql.json
"""

import os
import re
import requests
from dotenv import load_dotenv
from typing import List
from rapidfuzz import process, fuzz

load_dotenv()

# ─────────────────────────────────────────────
# Shopify Config
# ─────────────────────────────────────────────

SHOPIFY_STORE_URL = os.getenv("SHOPIFY_STORE_URL", "")
SHOPIFY_ACCESS_TOKEN = os.getenv("X_SHOPIFY_ACCESS_TOKEN", "")

GRAPHQL_URL = f"https://{SHOPIFY_STORE_URL}/admin/api/2026-01/graphql.json"

SHOPIFY_HEADERS = {
    "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN,
    "Content-Type": "application/json",
}

VALID_TAGS = [
    "Wallet", "Handbags", "Card Holder", "Bags",
    "Gifts", "Accessories", "Ladies Wallet", "Travel", "featured collection"
]


# ─────────────────────────────────────────────
# Core GraphQL Executor
# ─────────────────────────────────────────────

def gql(query: str, variables: dict = None) -> dict:
    """
    Execute a GraphQL query against the Shopify Admin API.

    Args:
        query: GraphQL query string.
        variables: Optional dict of variables referenced in the query.

    Returns:
        The 'data' portion of the GraphQL response.

    Raises:
        RuntimeError if the HTTP request fails or GraphQL returns errors.
    """
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    try:
        response = requests.post(GRAPHQL_URL, json=payload, headers=SHOPIFY_HEADERS, timeout=20)
        response.raise_for_status()
        result = response.json()

        if "errors" in result:
            raise RuntimeError(f"GraphQL errors: {result['errors']}")

        return result.get("data", {})

    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Shopify GraphQL HTTP error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Shopify GraphQL request failed: {str(e)}")


def gql_paginated(query: str, variables: dict, data_path: list, max_pages: int = 5) -> list:
    """
    Execute a paginated GraphQL query using Shopify's cursor-based pagination.

    The query MUST:
    - Accept a $cursor variable (String)
    - Return pageInfo { hasNextPage endCursor } on the connection
    - Return edges { node { ... } } on the connection

    Args:
        query: GraphQL query string with $cursor variable.
        variables: Initial variables dict (cursor will be managed automatically).
        data_path: List of keys to traverse from 'data' to the connection object.
                   e.g. ["orders"] reaches data.orders
                   e.g. ["products"] reaches data.products
        max_pages: Safety cap (default 5 = up to 1,250 records at 250/page).

    Returns:
        Flat list of all node dicts collected across all pages.
    """
    all_nodes = []
    cursor = None
    page = 0

    while page < max_pages:
        vars_with_cursor = {**variables, "cursor": cursor}
        data = gql(query, vars_with_cursor)

        # Traverse data_path to reach the connection object
        connection = data
        for key in data_path:
            connection = connection.get(key, {})

        nodes = [edge["node"] for edge in connection.get("edges", [])]
        all_nodes.extend(nodes)

        page_info = connection.get("pageInfo", {})
        if not page_info.get("hasNextPage", False):
            break

        cursor = page_info.get("endCursor")
        page += 1

    return all_nodes


# ─────────────────────────────────────────────
# Formatting Helpers
# ─────────────────────────────────────────────

def format_money(amount) -> str:
    """
    Format a numeric or string amount as Pakistani Rupees (PKR).
    Handles None, empty string, and float/string inputs gracefully.
    """
    try:
        if amount is None or amount == "":
            return "Rs. 0.00"
        return f"Rs. {float(amount):,.2f}"
    except (ValueError, TypeError):
        return str(amount)


# ─────────────────────────────────────────────
# Product Filtering Helpers
# ─────────────────────────────────────────────

def filter_products_by_price(products: list, max_price: float) -> list:
    """
    Filter summarized products where at least one variant price is <= max_price (PKR).

    Works on already-summarized product dicts (output of summarize_product).

    Args:
        products: List of summarized product dicts.
        max_price: Maximum price in PKR.

    Returns:
        Filtered list of products.
    """
    result = []
    for p in products:
        variant_prices = []
        for v in p.get("variants", []):
            raw = v.get("price", "").replace("Rs.", "").replace(",", "").strip()
            try:
                variant_prices.append(float(raw))
            except ValueError:
                pass
        if variant_prices and min(variant_prices) <= max_price:
            result.append(p)
    return result

def filter_products_by_name(products: list, name_query: str) -> list:
    """
    Filter products by fuzzy name match against the product title.

    Allows matching even when the user doesn't type the exact product name.
    Example: "leather slim wallet" will match "Men's Slim Leather Wallet".

    Args:
        products: List of product dicts.
        name_query: The user's approximate search string.

    Returns:
        Filtered list sorted by match confidence (best matches first).
    """
    if not name_query:
        return products

    query_strip = name_query.strip()

    if not query_strip:
        return products
    
    titles = [p.get("title", "") for p in products if p.get("title")]
    
    exact_matches = is_same_string(query_strip, titles)
    
    if exact_matches:
        return [p for p in products if p.get("title") == exact_matches[0]]
    
    matched = set()
    fuzzy_matches = process.extract(
            query_strip,
            titles,
            scorer=fuzz.WRatio,
            processor=str.lower,
            score_cutoff=65,
            limit=2
        )
    matched.update(m[0] for m in fuzzy_matches)
    
    if not matched:
        return "No matching product found. Please refine."

    if len(matched) > 1:
        return f"product with closely similar names found {"[" + ", ".join(f'"{m}"' for m in matched) + "]"}. Which of these products is user looking for."

    # Single fuzzy match
    match = next(iter(matched))
    return f"product with closely similar name found {'"' + match + '"'}. Re-infer this"


def is_same_string(a: str, b: List) -> List[str]:
    """
    Compare all values in `a` with all values in `b` (after normalization).
    Return a list of unique original `b` values that match.
    """
    # print("inside is_same_string: ", a, b)
    def normalize(s: str) -> str:
        # Remove all non-alphanumeric characters and lowercase
        return re.sub(r"[^a-zA-Z0-9]", "", s).lower()

    # Track unique matches from original b
    matched_b = set()

    # Compare all combinations
    for b_item in b:
        if normalize(a) == normalize(b_item):
            # print(f"Exact match found: '{normalize(a)}' == '{normalize(b_item)}'")
            matched_b.add(b_item)

    # Return as list
    return list(matched_b)

# ─────────────────────────────────────────────
# Product Summarizer
# ─────────────────────────────────────────────

def summarize_product(product: dict) -> dict:
    """
    Convert a raw GraphQL product node into a compact, LLM-friendly summary.

    Expects the GraphQL product node shape:
        id, title, tags (list), description,
        variants { edges { node { title, price, inventoryQuantity, sku } } }

    Returns:
        Dict: id, title, tags (str), price_range (PKR), in_stock, variants list, description.
    """
    variants_raw = [
        edge["node"]
        for edge in product.get("variants", {}).get("edges", [])
    ]

    prices = []
    inventories = []
    variant_summary = []

    for v in variants_raw:
        price_val = v.get("price", "0") or "0"
        inv = v.get("inventoryQuantity", 0) or 0
        try:
            prices.append(float(price_val))
        except ValueError:
            prices.append(0.0)
        inventories.append(inv)
        variant_summary.append({
            "title": v.get("title", "Default"),
            "price": format_money(price_val),
            "inventory": inv,
            "sku": v.get("sku", ""),
        })

    price_range = ""
    if prices:
        mn, mx = min(prices), max(prices)
        price_range = format_money(mn) if mn == mx else f"{format_money(mn)} - {format_money(mx)}"

    in_stock = any(q > 0 for q in inventories)

    # GraphQL returns tags as a list
    tags = product.get("tags", [])
    tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)

    return {
        "id": product.get("id", ""),
        "title": product.get("title", ""),
        "tags": tags_str,
        "price_range": price_range,
        "in_stock": in_stock,
        "variants": variant_summary,
        "description": (product.get("description", "") or "")[:300],
    }


# ─────────────────────────────────────────────
# Order Summarizer
# ─────────────────────────────────────────────

def summarize_order(order: dict) -> dict:
    """
    Convert a raw GraphQL order node into a compact, LLM-friendly summary.

    Expects GraphQL order node with fields from ORDER_FIELDS defined below.

    Returns:
        Dict with order metadata, line items, fulfillment tracking, and refunds.
    """
    # Line items
    line_items = [
        {
            "title": edge["node"].get("title"),
            "quantity": edge["node"].get("quantity"),
            "price": format_money(edge["node"].get("originalUnitPrice")),
        }
        for edge in order.get("lineItems", {}).get("edges", [])
    ]

    # Fulfillments
    fulfillments = []
    for f in order.get("fulfillments", []):
        tracking_info = f.get("trackingInfo") or []
        fulfillments.append({
            "status": f.get("status"),
            "tracking_number": tracking_info[0].get("number") if tracking_info else None,
            "tracking_url": tracking_info[0].get("url") if tracking_info else None,
        })

    # Refunds
    refunds = []
    for r in order.get("refunds", []):
        transactions = [
            {
                "amount": format_money(
                    t["node"].get("amountSet", {}).get("shopMoney", {}).get("amount")
                ),
                "status": t["node"].get("status"),
            }
            for t in r.get("transactions", {}).get("edges", [])
        ]
        refunds.append({
            "created_at": r.get("createdAt"),
            "note": r.get("note"),
            "transactions": transactions,
        })

    return {
        "id": order.get("id", ""),
        "name": order.get("name", ""),
        "email": order.get("email", ""),
        "created_at": order.get("createdAt", ""),
        "financial_status": order.get("displayFinancialStatus", ""),
        "fulfillment_status": order.get("displayFulfillmentStatus", ""),
        "total_price": format_money(
            order.get("totalPriceSet", {}).get("shopMoney", {}).get("amount", "0")
        ),
        "line_items": line_items,
        "shipping_address": order.get("shippingAddress"),
        "fulfillments": fulfillments,
        "refunds": refunds,
        "tags": order.get("tags", []),
    }


# ─────────────────────────────────────────────
# Reusable GraphQL Field Blocks
# ─────────────────────────────────────────────

# Product fields — used in product queries
PRODUCT_FIELDS = """
    id
    title
    tags
    description
    variants(first: 20) {
        edges {
            node {
                title
                price
                sku
                inventoryQuantity
            }
        }
    }
"""

# Order fields — used in order queries
ORDER_FIELDS = """
    id
    name
    email
    createdAt
    note
    tags
    displayFinancialStatus
    displayFulfillmentStatus
    totalPriceSet {
        shopMoney { amount }
    }
    shippingAddress {
        firstName
        lastName
        address1
        city
        country
        phone
    }
    lineItems(first: 20) {
        edges {
            node {
                title
                quantity
                originalUnitPrice
            }
        }
    }
    fulfillments {
        status
        trackingInfo {
            number
            url
        }
    }
    refunds {
        createdAt
        note
        transactions(first: 5) {
            edges {
                node {
                    status
                    amountSet {
                        shopMoney { amount }
                    }
                }
            }
        }
    }
"""