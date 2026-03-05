"""
utils.py — Shared utilities for the Silk Skin AI Agent system.

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

    Returns the 'data' portion of the response.
    Raises RuntimeError on HTTP failure or GraphQL errors.
    """
    payload = {"query": query, **({"variables": variables} if variables else {})}
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
        raise RuntimeError(f"Shopify GraphQL request failed: {e}")


def gql_paginated(query: str, variables: dict, data_path: list, max_pages: int = 5) -> list:
    """
    Execute a cursor-paginated GraphQL query.

    The query must accept $cursor (String) and return:
        pageInfo { hasNextPage endCursor }
        edges { node { ... } }

    Args:
        query:      GraphQL query string with $cursor variable.
        variables:  Initial variables (cursor managed automatically).
        data_path:  Keys to traverse from 'data' to the connection (e.g. ["orders"]).
        max_pages:  Page cap — default 5 (up to 1,250 records at 250/page).

    Returns:
        Flat list of all node dicts across all pages.
    """
    all_nodes, cursor, page = [], None, 0
    while page < max_pages:
        data = gql(query, {**variables, "cursor": cursor})
        connection = data
        for key in data_path:
            connection = connection.get(key, {})
        all_nodes.extend(edge["node"] for edge in connection.get("edges", []))
        page_info = connection.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        page += 1
    return all_nodes


# ─────────────────────────────────────────────
# Formatting Helpers
# ─────────────────────────────────────────────

def format_money(amount) -> str:
    """Format a value as PKR. Handles None, empty string, float, and string inputs."""
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

    Args:
        products:   List of summarized product dicts (output of summarize_product).
        max_price:  Maximum price in PKR.
    """
    result = []
    for p in products:
        prices = []
        for v in p.get("variants", []):
            raw = v.get("price", "").replace("Rs.", "").replace(",", "").strip()
            try:
                prices.append(float(raw))
            except ValueError:
                pass
        if prices and min(prices) <= max_price:
            result.append(p)
    return result


def filter_products_by_name(products: list, name_query: str) -> list | str:
    """
    Filter products by fuzzy name match.

    Returns:
        - list of matched product dicts on a confident single match.
        - str disambiguation message if multiple close matches exist.
        - str not-found message if no match meets the confidence threshold.
    """
    if not name_query or not name_query.strip():
        return products

    query_strip = name_query.strip()
    titles = [p.get("title", "") for p in products if p.get("title")]

    # Exact (normalized) match takes priority
    exact = is_same_string(query_strip, titles)
    if exact:
        return [p for p in products if p.get("title") == exact[0]]

    fuzzy_matches = process.extract(
        query_strip, titles,
        scorer=fuzz.WRatio, processor=str.lower,
        score_cutoff=65, limit=2,
    )
    matched = {m[0] for m in fuzzy_matches}

    if not matched:
        return "No matching product found. Please refine your search."
    if len(matched) > 1:
        options = ", ".join(f'"{m}"' for m in matched)
        return f"Multiple similar products found: [{options}]. Which one did you mean?"

    match = next(iter(matched))
    return [p for p in products if p.get("title") == match]


def is_same_string(a: str, b: List[str]) -> List[str]:
    """Return items from b that normalize-match a (strips non-alphanumeric, lowercased)."""
    def normalize(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]", "", s).lower()
    return [item for item in b if normalize(a) == normalize(item)]


# ─────────────────────────────────────────────
# Product Summarizer
# ─────────────────────────────────────────────

def summarize_product(product: dict) -> dict:
    """
    Convert a raw GraphQL product node into a compact, LLM-friendly summary.

    Expects: id, title, tags (list), description,
             variants { edges { node { title, price, inventoryQuantity } } }

    Returns:
        Dict with keys: title, tags (str), price_range (PKR), in_stock,
        description, and variants — each variant has: title, price (PKR str),
        inventory_quantity (int), sku (str).
    """
    variants_raw = [edge["node"] for edge in product.get("variants", {}).get("edges", [])]
    prices, inventories, variant_summary = [], [], []

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
            "inventory_quantity": inv,
            "sku": v.get("sku", ""),
        })

    if prices:
        mn, mx = min(prices), max(prices)
        price_range = format_money(mn) if mn == mx else f"{format_money(mn)} - {format_money(mx)}"
    else:
        price_range = ""

    tags = product.get("tags", [])
    return {
        "title": product.get("title", ""),
        "tags": ", ".join(tags) if isinstance(tags, list) else str(tags),
        "price_range": price_range,
        "in_stock": any(q > 0 for q in inventories),
        "variants": variant_summary,
        "description": (product.get("description", "") or "")[:300],
    }


# ─────────────────────────────────────────────
# Order Summarizer
# ─────────────────────────────────────────────

def summarize_order(order: dict) -> dict:
    """
    Convert a raw GraphQL order node into a compact, LLM-friendly summary.

    Returns:
        Dict with order metadata, line items, fulfillment tracking, and refunds.
    """
    line_items = [
        {
            "title": edge["node"].get("title"),
            "quantity": edge["node"].get("quantity"),
            "price": format_money(edge["node"].get("originalUnitPrice")),
        }
        for edge in order.get("lineItems", {}).get("edges", [])
    ]

    fulfillments = []
    for f in order.get("fulfillments", []):
        tracking = f.get("trackingInfo") or []
        fulfillments.append({
            "status": f.get("status"),
            "tracking_number": tracking[0].get("number") if tracking else None,
            "tracking_url": tracking[0].get("url") if tracking else None,
        })

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