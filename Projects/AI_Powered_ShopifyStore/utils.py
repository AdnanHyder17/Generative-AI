"""
utils.py - Shared utilities for the Shopify AI Agent system.

Covers:
  - Shopify REST API client with error handling
  - Date/time helpers
  - Response formatters
  - Logging setup
"""

import logging
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import requests

from config.settings import settings

# ─── Logging ────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger("utils")


# ─── Shopify API Client ──────────────────────────────────────────────────────

class ShopifyClient:
    """
    Thin wrapper around Shopify Admin REST API.
    All methods return plain dicts / lists and raise on HTTP errors.
    """

    def __init__(self):
        self.base_url = settings.shopify_base_url
        self.headers = settings.shopify_headers
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}/{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "unknown"
            body = e.response.text if e.response else ""
            raise RuntimeError(
                f"Shopify GET {endpoint} failed [{status}]: {body}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error on GET {endpoint}: {e}") from e

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}/{endpoint}"
        try:
            resp = self.session.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "unknown"
            body = e.response.text if e.response else ""
            raise RuntimeError(
                f"Shopify POST {endpoint} failed [{status}]: {body}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error on POST {endpoint}: {e}") from e

    # ── Products ──────────────────────────────────────────────────────────────

    def search_products(
        self,
        query: str = "",
        product_type: str = "",
        limit: int = 10,
    ) -> list[dict]:
        params: dict = {"limit": limit}
        if query:
            params["title"] = query
        if product_type:
            params["product_type"] = product_type
        data = self._get("products.json", params=params)
        return data.get("products", [])

    def get_product(self, product_id: int) -> dict:
        data = self._get(f"products/{product_id}.json")
        return data.get("product", {})

    def get_product_variants(self, product_id: int) -> list[dict]:
        product = self.get_product(product_id)
        return product.get("variants", [])

    def get_all_products(self, limit: int = 250) -> list[dict]:
        data = self._get("products.json", params={"limit": limit})
        return data.get("products", [])

    # ── Collections ───────────────────────────────────────────────────────────

    def get_collections(self) -> list[dict]:
        data = self._get("custom_collections.json")
        return data.get("custom_collections", [])

    def get_collection_products(self, collection_id: int, limit: int = 20) -> list[dict]:
        data = self._get(
            "products.json",
            params={"collection_id": collection_id, "limit": limit},
        )
        return data.get("products", [])

    # ── Orders ────────────────────────────────────────────────────────────────

    def get_order(self, order_id: int) -> dict:
        data = self._get(f"orders/{order_id}.json")
        return data.get("order", {})

    def get_orders(self, params: Optional[dict] = None) -> list[dict]:
        data = self._get("orders.json", params=params or {"status": "any", "limit": 250})
        return data.get("orders", [])

    def get_orders_in_date_range(
        self, created_at_min: str, created_at_max: str, status: str = "any"
    ) -> list[dict]:
        params = {
            "created_at_min": created_at_min,
            "created_at_max": created_at_max,
            "status": status,
            "limit": 250,
        }
        return self.get_orders(params)

    def get_unfulfilled_orders(self) -> list[dict]:
        return self.get_orders(
            {"fulfillment_status": "unfulfilled", "status": "open", "limit": 250}
        )

    def get_refunded_orders(self, created_at_min: str) -> list[dict]:
        orders = self.get_orders(
            {
                "status": "any",
                "created_at_min": created_at_min,
                "limit": 250,
            }
        )
        return [o for o in orders if o.get("refunds")]

    # ── Inventory ─────────────────────────────────────────────────────────────

    def get_inventory_levels(self) -> list[dict]:
        data = self._get("inventory_levels.json", params={"limit": 250})
        return data.get("inventory_levels", [])

    # ── Customers ─────────────────────────────────────────────────────────────

    def get_customers(self, limit: int = 250) -> list[dict]:
        data = self._get("customers.json", params={"limit": limit})
        return data.get("customers", [])

    def get_customer(self, customer_id: int) -> dict:
        data = self._get(f"customers/{customer_id}.json")
        return data.get("customer", {})


# ─── Date Helpers ────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_days_ago(days: int) -> str:
    """Return ISO-8601 UTC timestamp for N days ago."""
    return (utc_now() - timedelta(days=days)).isoformat()


def start_of_today_iso() -> str:
    now = utc_now()
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def start_of_month_iso() -> str:
    now = utc_now()
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()


def start_of_last_month_iso() -> str:
    now = utc_now()
    first_of_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_month = first_of_this - timedelta(days=1)
    return last_month.replace(day=1).isoformat()


def end_of_last_month_iso() -> str:
    now = utc_now()
    first_of_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_day = first_of_this - timedelta(seconds=1)
    return last_day.isoformat()


# ─── Response Formatters ─────────────────────────────────────────────────────

def format_price(amount: Any) -> str:
    """Format a numeric string or float as currency."""
    try:
        return f"${float(amount):.2f}"
    except (TypeError, ValueError):
        return str(amount)


def format_product_summary(product: dict) -> str:
    """One-line product summary for LLM context."""
    title = product.get("title", "Unknown")
    variants = product.get("variants", [])
    prices = [v.get("price", "0") for v in variants]
    min_price = min(prices, key=lambda x: float(x or 0)) if prices else "N/A"
    tags = product.get("tags", "")
    status = product.get("status", "unknown")
    return (
        f"• {title} | From {format_price(min_price)} | "
        f"Status: {status} | Tags: {tags}"
    )


def format_order_summary(order: dict) -> str:
    """One-line order summary for LLM context."""
    name = order.get("name", f"#{order.get('id', '?')}")
    total = format_price(order.get("total_price", "0"))
    fulfillment = order.get("fulfillment_status") or "unfulfilled"
    financial = order.get("financial_status", "unknown")
    created = order.get("created_at", "")[:10]
    customer_name = "Guest"
    if order.get("customer"):
        c = order["customer"]
        customer_name = f"{c.get('first_name','')} {c.get('last_name','')}".strip() or "Guest"
    return (
        f"Order {name} | {created} | {customer_name} | "
        f"{total} | Fulfillment: {fulfillment} | Payment: {financial}"
    )


def calculate_total_sales(orders: list[dict]) -> float:
    """Sum total_price from a list of orders."""
    total = 0.0
    for o in orders:
        try:
            total += float(o.get("total_price", 0) or 0)
        except (TypeError, ValueError):
            pass
    return total


def calculate_average_order_value(orders: list[dict]) -> float:
    if not orders:
        return 0.0
    return calculate_total_sales(orders) / len(orders)


def extract_product_sales(orders: list[dict]) -> dict[str, dict]:
    """
    Aggregate quantity sold and revenue per product title from orders.
    Returns {product_title: {quantity, revenue}}
    """
    sales: dict[str, dict] = {}
    for order in orders:
        for item in order.get("line_items", []):
            title = item.get("title", "Unknown")
            qty = int(item.get("quantity", 0))
            price = float(item.get("price", 0) or 0)
            if title not in sales:
                sales[title] = {"quantity": 0, "revenue": 0.0}
            sales[title]["quantity"] += qty
            sales[title]["revenue"] += qty * price
    return sales


def pretty_json(data: Any) -> str:
    """Pretty-print JSON for LLM tool results."""
    return json.dumps(data, indent=2, default=str)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_order_id_from_text(text: str) -> Optional[int]:
    """Pull a numeric order ID from user text like '#45821' or 'order 45821'."""
    match = re.search(r"#?(\d{4,})", text)
    return int(match.group(1)) if match else None