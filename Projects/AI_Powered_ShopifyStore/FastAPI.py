"""
FastAPI server for stores frontend chatbot widget.

Usage:
    uvicorn FastAPI:app --host 0.0.0.0 --port 8000
"""

import json
from graph import graph
from shopify_proxy import router as proxy_router
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

app = FastAPI()

app.include_router(proxy_router)

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
The frontend widget adding part in layout/theme.liquid in shopify edit theme (provide me updated version of this code change anything you like) 
call appropriate API endpoints to fetch real data for the workflows instead of placeholders.

<script>
(function () {
  /* ── Config ──────────────────────────────────────────── */
  const WS_URL   = 'ws://localhost:8000/ws/chat' // 'wss://YOUR_SERVER_DOMAIN/ws/chat'; ← Update this
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