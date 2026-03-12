"""
Shopify Chat Proxy Routes
Add these endpoints to your existing FastAPI app (FastAPI.py).

These are called directly from the frontend widget to fetch real Shopify data
without exposing your access token in the browser.

Mount prefix: /apps/silkskin-chat
"""

import os
import httpx
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/apps/silkskin-chat")

SHOPIFY_STORE   = os.getenv("SHOPIFY_STORE_URL", "silkskinonline.myshopify.com")
SHOPIFY_TOKEN   = os.getenv("X_SHOPIFY_ACCESS_TOKEN")
SHOPIFY_VERSION = os.getenv("SHOPIFY_API_VERSION", "2026-01")

SHOPIFY_BASE = f"https://{SHOPIFY_STORE}/admin/api/{SHOPIFY_VERSION}"
HEADERS      = {
    "X-Shopify-Access-Token": SHOPIFY_TOKEN,
    "Content-Type": "application/json",
}


# ── Order Status ───────────────────────────────────────────────────────────────
@router.get("/order-status")
async def order_status(
    order: str = Query(..., description="Order number without #"),
    email: str = Query(..., description="Email on the order for verification"),
):
    """
    Lookup an order by number and verify with the customer's email.
    Returns a safe subset of order data — no PII beyond what the customer knows.
    """
    if not SHOPIFY_TOKEN:
        raise HTTPException(status_code=500, detail="Shop not configured")

    # Normalise: strip # and whitespace
    order_number = order.lstrip("#").strip()
    email_lower  = email.lower().strip()

    async with httpx.AsyncClient(timeout=10) as client:
        # Search by order name (Shopify uses "name" = "#1234")
        resp = await client.get(
            f"{SHOPIFY_BASE}/orders.json",
            headers=HEADERS,
            params={
                "name": f"#{order_number}",
                "status": "any",
                "fields": "id,name,email,financial_status,fulfillment_status,fulfillments,updated_at",
            },
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Shopify API error")

    orders = resp.json().get("orders", [])
    if not orders:
        return JSONResponse({"error": "not_found"})

    order_obj = orders[0]

    # Verify email
    if order_obj.get("email", "").lower() != email_lower:
        return JSONResponse({"error": "email_mismatch"})

    # Extract tracking info from most recent fulfillment
    tracking_number = None
    tracking_url    = None
    carrier         = None
    fulfillments    = order_obj.get("fulfillments", [])
    if fulfillments:
        latest = fulfillments[-1]
        tracking_number = latest.get("tracking_number")
        tracking_url    = latest.get("tracking_url")
        carrier         = latest.get("tracking_company")

    return {
        "name":               order_obj["name"],
        "financial_status":   order_obj.get("financial_status"),
        "fulfillment_status": order_obj.get("fulfillment_status"),
        "tracking_number":    tracking_number,
        "tracking_url":       tracking_url,
        "carrier":            carrier,
        "updated_at":         order_obj.get("updated_at"),
    }


# ── Active Discounts ───────────────────────────────────────────────────────────
@router.get("/discounts")
async def active_discounts():
    """
    Returns currently active price-rule based discount codes.
    Filters to those that are started, not yet expired, and still have usage remaining.
    """
    if not SHOPIFY_TOKEN:
        raise HTTPException(status_code=500, detail="Shop not configured")

    now = datetime.now(timezone.utc)

    async with httpx.AsyncClient(timeout=10) as client:
        # Fetch price rules
        resp = await client.get(
            f"{SHOPIFY_BASE}/price_rules.json",
            headers=HEADERS,
            params={"limit": 50},
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Shopify API error")

    rules = resp.json().get("price_rules", [])
    result = []

    async with httpx.AsyncClient(timeout=10) as client:
        for rule in rules:
            # Skip if not started yet
            starts = rule.get("starts_at")
            if starts and datetime.fromisoformat(starts.replace("Z", "+00:00")) > now:
                continue
            # Skip if expired
            ends = rule.get("ends_at")
            if ends and datetime.fromisoformat(ends.replace("Z", "+00:00")) < now:
                continue
            # Skip if usage maxed out
            usage_limit = rule.get("usage_limit")
            usage_count = rule.get("usage_count", 0)
            if usage_limit and usage_count >= usage_limit:
                continue

            # Fetch discount codes for this rule
            codes_resp = await client.get(
                f"{SHOPIFY_BASE}/price_rules/{rule['id']}/discount_codes.json",
                headers=HEADERS,
                params={"limit": 5},
            )
            if codes_resp.status_code != 200:
                continue

            codes = codes_resp.json().get("discount_codes", [])
            if not codes:
                continue

            # Build human-readable description
            value_type = rule.get("value_type")           # "percentage" | "fixed_amount"
            value      = abs(float(rule.get("value", 0)))
            if value_type == "percentage":
                desc = f"{int(value)}% off your order"
            elif value_type == "fixed_amount":
                desc = f"${value:.0f} off your order"
            else:
                desc = rule.get("title", "Special discount")

            min_req = rule.get("prerequisite_subtotal_range")
            if min_req:
                min_val = min_req.get("greater_than_or_equal_to", 0)
                desc += f" (min. ${float(min_val):.0f} spend)"

            for code_obj in codes:
                result.append({
                    "code":     code_obj["code"],
                    "title":    desc,
                    "ends_at":  ends,
                })

    return {"discounts": result[:8]}  # Cap at 8 for UI clarity


# ── Mount this router in FastAPI.py ───────────────────────────────────────────
# In your FastAPI.py, add:
#
#   from shopify_proxy import router as proxy_router
#   app.include_router(proxy_router)
#
# And update CORS to allow your storefront origin.