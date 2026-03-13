"""
FastAPI server for stores frontend chatbot widget.

Usage:
    uvicorn FastAPI:app --host 0.0.0.0 --port 8000
"""

import os
import json
import httpx
from graph import graph
from datetime import datetime, timezone
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter, HTTPException, Query

app = FastAPI()

"""
Shopify Chat Proxy Routes
Add these endpoints to your existing FastAPI app (FastAPI.py).

These are called directly from the frontend widget to fetch real Shopify data
without exposing your access token in the browser.

Mount prefix: /apps/silkskin-chat
"""

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


app.include_router(router)

# 1. FIX CORS: This allows your Shopify store to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://silkskinonline.myshopify.com"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/chat")
async def websocket_endpoint(
    websocket: WebSocket, 
    role: str = Query(...),   # "admin" or "customer"
    user_id: str = Query(...) # Shopify Customer ID or Admin ID
):
    await websocket.accept()
    print(f"New connection: Role={role}, ID={user_id}")

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_msg = message_data.get("message")
            
            config = {"configurable": {"thread_id": user_id}}
            state_input = {
                "messages": [HumanMessage(content=user_msg)],
                "user_role": role,
            }

            try:
                result = graph.invoke(state_input, config=config)
                messages = result.get("messages", [])
                
                ai_response = "I'm sorry, I couldn't generate a response."
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        content = msg.content
                        if isinstance(content, list):
                            content = " ".join(
                                b.get("text", "") for b in content 
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        ai_response = content
                        break
            except Exception as e:
                ai_response = f"Error processing request: {str(e)}"

            await websocket.send_text(json.dumps({
                "reply": ai_response,
                "role": role # Optional: echo back the role
            }))
    except WebSocketDisconnect:
        print(f"User {user_id} disconnected.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 

    
"""
The frontend code added in layout/theme.liquid in shopify edit theme for chat widget.

<script>
(function () {
  /* ── Config ──────────────────────────────────────────── */
  const WS_URL   = 'ws://localhost:8000/ws/chat'; // Update if your API is hosted elsewhere
  const SHOP_URL = 'https://silkskinonline.myshopify.com';
  const API_VER  = '2026-01';

  /* ── State ───────────────────────────────────────────── */
  let socket       = null;
  let isWaiting    = false;
  let chatMode     = 'menu';   // 'menu' | 'track' | 'chat'
  let trackStep    = 0;        // 0=idle, 1=awaiting order#, 2=awaiting email
  let pendingOrder = null;
  let isOpen       = false;
  let unread       = 0;

  /* ── DOM ─────────────────────────────────────────────── */
  const widget    = document.getElementById('ss-widget');
  const fab       = document.getElementById('ss-fab');
  const win       = document.getElementById('ss-window');
  const msgs      = document.getElementById('ss-messages');
  const input     = document.getElementById('ss-input');
  const sendBtn   = document.getElementById('ss-send');
  const sendIcon  = sendBtn.querySelector('.ss-send-icon');
  const sendSpin  = document.getElementById('ss-spin');
  const typing    = document.getElementById('ss-typing');
  const badge     = document.getElementById('ss-badge');

  /* ── Toggle ──────────────────────────────────────────── */
  function ssToggle() {
    isOpen = !isOpen;
    widget.classList.toggle('open', isOpen);
    if (isOpen) {
      clearBadge();
      if (msgs.children.length === 0) showWelcome();
      setTimeout(() => input.focus(), 350);
      initSocket();
    }
  }
  window.ssToggle = ssToggle;
  fab.onclick = ssToggle;

  function showBadge(n) {
    unread += n;
    badge.textContent = unread > 9 ? '9+' : unread;
    badge.classList.add('show');
  }
  function clearBadge() {
    unread = 0;
    badge.classList.remove('show');
  }

  /* ── WebSocket ───────────────────────────────────────── */
  function initSocket() {
    if (socket && socket.readyState < 2) return;
    const userId = '{{ customer.id | default: "guest_" | append: "session" }}';
    const role   = '{{ customer.id }}' ? 'customer' : 'guest';
    try {
      socket = new WebSocket(`${WS_URL}?role=${role}&user_id=${userId}`);
      socket.onmessage = (e) => {
        setLoading(false);
        const data = JSON.parse(e.data);
        appendBotMsg(data.reply);
        if (!isOpen) showBadge(1);
      };
      socket.onerror = () => {
        setLoading(false);
        appendBotMsg("Connection issue — please try again shortly.");
      };
    } catch(err) { console.warn('[SS Chat] WebSocket failed:', err); }
  }

  /* ── Messages ────────────────────────────────────────── */
  function appendBotMsg(text) {
    const wrap = document.createElement('div');
    wrap.className = 'ss-bubble-wrap bot';
    const bub  = document.createElement('div');
    bub.className = 'ss-bubble bot';
    bub.textContent = text;
    wrap.appendChild(bub);
    msgs.appendChild(wrap);
    scrollBottom();
  }

  function appendUserMsg(text) {
    const wrap = document.createElement('div');
    wrap.className = 'ss-bubble-wrap user';
    const bub  = document.createElement('div');
    bub.className = 'ss-bubble user';
    bub.textContent = text;
    wrap.appendChild(bub);
    msgs.appendChild(wrap);
    scrollBottom();
  }

  function appendCard(el) {
    const wrap = document.createElement('div');
    wrap.className = 'ss-bubble-wrap bot';
    wrap.appendChild(el);
    msgs.appendChild(wrap);
    scrollBottom();
  }

  function appendActions(actions) {
    // actions = [{icon, label, cb}]
    const wrap = document.createElement('div');
    wrap.className = 'ss-bubble-wrap bot';
    const bub = document.createElement('div');
    bub.className = 'ss-bubble bot';

    const grp = document.createElement('div');
    grp.className = 'ss-actions';
    actions.forEach(a => {
      const btn = document.createElement('button');
      btn.className = 'ss-action-btn';
      btn.innerHTML = `<span class="ss-btn-icon">${a.icon}</span>${a.label}`;
      btn.onclick = () => { btn.closest('.ss-bubble-wrap').remove(); a.cb(); };
      grp.appendChild(btn);
    });
    bub.appendChild(grp);
    wrap.appendChild(bub);
    msgs.appendChild(wrap);
    scrollBottom();
  }

  function scrollBottom() {
    msgs.scrollTop = msgs.scrollHeight;
  }

  /* ── Loading State ───────────────────────────────────── */
  function setLoading(on) {
    isWaiting = on;
    typing.classList.toggle('show', on);
    input.disabled = on;
    sendBtn.disabled = on;
    sendIcon.style.display = on ? 'none' : '';
    sendSpin.style.display  = on ? 'block' : 'none';
    scrollBottom();
  }

  /* ── Welcome ─────────────────────────────────────────── */
  function showWelcome() {
    const name = '{{ customer.first_name }}';
    appendBotMsg(name
      ? `Welcome back, ${name}! 👋 How can I help you today?`
      : "Hello! Welcome to Silk Skin 🌿 How can I help you today?"
    );
    showMainMenu();
  }

  function showMainMenu() {
    chatMode = 'menu';
    appendActions([
      { icon: '📦', label: 'Track My Order',    cb: startTrackOrder },
      { icon: '🏷️', label: 'Current Discounts', cb: loadDiscounts },
      { icon: '💬', label: 'Talk to Assistant', cb: startChat },
    ]);
  }

  /* ────────────────────────────────────────────────────── */
  /*  WORKFLOW 1: Track Order                               */
  /* ────────────────────────────────────────────────────── */
  function startTrackOrder() {
    chatMode = 'track'; trackStep = 1;
    appendUserMsg('📦 Track My Order');
    appendBotMsg('Please enter your order number (e.g. #1234):');
    input.placeholder = 'Order number…';
    input.focus();
  }

  async function handleTrackInput(val) {
    if (trackStep === 1) {
      // Normalise order number
      const num = val.replace(/[^0-9]/g, '');
      if (!num) { appendBotMsg('That doesn\'t look like a valid order number. Try again (e.g. #1234):'); return; }
      pendingOrder = num;
      trackStep = 2;
      appendUserMsg(val);
      appendBotMsg('Got it! Please enter the email address on the order to verify:');
      input.placeholder = 'Email address…';
    } else if (trackStep === 2) {
      appendUserMsg(val);
      setLoading(true);
      await fetchOrderStatus(pendingOrder, val.trim());
    }
  }

  async function fetchOrderStatus(orderNum, email) {
    try {
      // Use Shopify's storefront-safe order lookup
      // We proxy through a Shopify App Proxy endpoint to keep tokens server-side
      const res  = await fetch(`/apps/silkskin-chat/order-status?order=${encodeURIComponent(orderNum)}&email=${encodeURIComponent(email)}`);
      setLoading(false);
      
      const text = await res.text();

      appendBotMsg(`response body: ${text}`);
      appendBotMsg(`status: ${res.status}`);
      appendBotMsg(`ok: ${res.ok}`);
      appendBotMsg(`url: ${res.url}`);

      if (!res.ok) throw new Error('not_found');
      const order = await res.json();

      if (order.error) {
        appendBotMsg("I couldn't find an order matching those details. Please double-check and try again.");
      } else {
        renderOrderCard(order);
      }
    } catch (e) {
      setLoading(false);
      appendBotMsg(`I wasn't able to look that up right now. Please visit our " +
        "Order Status page or contact support. ${e.message}`);
    }

    // Reset & offer menu
    trackStep = 0; pendingOrder = null; chatMode = 'menu';
    input.placeholder = 'Type a message…';
    setTimeout(() => {
      appendBotMsg('Is there anything else I can help you with?');
      showMainMenu();
    }, 800);
  }

  function renderOrderCard(o) {
    // o: { name, financial_status, fulfillment_status, tracking_number, tracking_url, carrier, updated_at }
    const card = document.createElement('div');
    card.className = 'ss-order-card';

    const fulfilMap = { fulfilled:'fulfilled', unfulfilled:'unfulfilled', in_transit:'in_transit' };
    const statusLabel = { fulfilled:'Fulfilled ✓', unfulfilled:'Unfulfilled', in_transit:'In Transit 🚚',
                          null:'Processing', undefined:'Processing' };
    const fs = o.fulfillment_status;
    const pill = `<span class="ss-status-pill ${fulfilMap[fs]||'unfulfilled'}">${statusLabel[fs]||'Processing'}</span>`;

    const trackRow = o.tracking_url
      ? `<a class="ss-track-link" href="${o.tracking_url}" target="_blank">🔗 Track Shipment (${o.carrier||'Carrier'})</a>`
      : '';

    card.innerHTML = `
      <div class="ss-order-card-header">
        <span>Order ${o.name}</span>
        ${pill}
      </div>
      <div class="ss-order-card-body">
        <div class="ss-order-row">
          <span class="ss-order-label">Payment</span>
          <span class="ss-order-val">${o.financial_status ? o.financial_status.charAt(0).toUpperCase() + o.financial_status.slice(1) : '—'}</span>
        </div>
        ${o.tracking_number ? `
        <div class="ss-order-row">
          <span class="ss-order-label">Tracking #</span>
          <span class="ss-order-val" style="font-size:12px">${o.tracking_number}</span>
        </div>` : ''}
        ${o.updated_at ? `
        <div class="ss-order-row">
          <span class="ss-order-label">Last updated</span>
          <span class="ss-order-val">${new Date(o.updated_at).toLocaleDateString()}</span>
        </div>` : ''}
        ${trackRow}
      </div>`;
    appendCard(card);
  }

  /* ────────────────────────────────────────────────────── */
  /*  WORKFLOW 2: Discounts                                 */
  /* ────────────────────────────────────────────────────── */
  async function loadDiscounts() {
    appendUserMsg('🏷️ Current Discounts');
    setLoading(true);

    try {
      const res  = await fetch('/apps/silkskin-chat/discounts');
      setLoading(false);
      if (!res.ok) throw new Error();
      const data = await res.json();

      if (!data.discounts || data.discounts.length === 0) {
        appendBotMsg("There are no active promotions right now, but check back soon! 🌸");
      } else {
        appendBotMsg(`Here are our current promotions:`);
        renderDiscountCards(data.discounts);
      }
    } catch(e) {
      setLoading(false);
      appendBotMsg("Couldn't load discounts right now. Check our website for the latest deals!");
    }

    setTimeout(() => {
      appendBotMsg('Anything else I can help you with?');
      showMainMenu();
    }, 600);
  }

  function renderDiscountCards(discounts) {
    const card = document.createElement('div');
    card.className = 'ss-discount-card';

    discounts.forEach(d => {
      const item = document.createElement('div');
      item.className = 'ss-discount-item';

      const expText = d.ends_at
        ? `Expires ${new Date(d.ends_at).toLocaleDateString()}`
        : 'No expiry';

      item.innerHTML = `
        <div class="ss-discount-code">
          ${d.code}
          <button class="ss-copy-btn" onclick="ssCopy('${d.code}', this)">Copy</button>
        </div>
        <div class="ss-discount-desc">${d.title || ''}</div>
        <div class="ss-discount-exp">📅 ${expText}</div>`;
      card.appendChild(item);
    });
    appendCard(card);
  }

  window.ssCopy = function(code, btn) {
    navigator.clipboard.writeText(code).then(() => {
      btn.textContent = 'Copied!';
      setTimeout(() => btn.textContent = 'Copy', 2000);
    });
  };

  /* ────────────────────────────────────────────────────── */
  /*  WORKFLOW 3: Open AI Chat                              */
  /* ────────────────────────────────────────────────────── */
  function startChat() {
    chatMode = 'chat';
    appendUserMsg('💬 Talk to Assistant');
    appendBotMsg("I'm ready to help! Ask me anything about our products, ingredients, skincare routines, or anything else 🌿");
    input.placeholder = 'Ask me anything…';
    input.focus();
  }

  /* ── Send Handler ────────────────────────────────────── */
  function sendMessage() {
    const val = input.value.trim();
    if (!val || isWaiting) return;
    input.value = '';

    if (chatMode === 'track') {
      handleTrackInput(val);
      return;
    }

    // AI Chat
    appendUserMsg(val);
    setLoading(true);

    if (!socket || socket.readyState !== 1) {
      initSocket();
      // Retry once socket opens
      const t = setInterval(() => {
        if (socket && socket.readyState === 1) {
          clearInterval(t);
          socket.send(JSON.stringify({ message: val }));
        }
      }, 200);
      setTimeout(() => clearInterval(t), 6000);
    } else {
      socket.send(JSON.stringify({ message: val }));
    }
  }

  sendBtn.onclick = sendMessage;
  input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
})();
</script>
"""