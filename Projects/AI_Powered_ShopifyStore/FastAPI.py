"""
FastAPI server for stores frontend chatbot widget.

Usage:
    uvicorn FastAPI:app --host 0.0.0.0 --port 8000
"""

import json
from graph import graph
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

app = FastAPI()

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
                "active_agent": "customer_support_agent", 
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
{
  "addresses": [],
  "addresses_count": 5,
  "b2b?": false,
  "company_available_locations": [],
  "company_available_locations_count": 1,
  "current_company": {},
  "current_location": null,
  "default_address": {},
  "email": "cornelius.potionmaker@gmail.com",
  "first_name": "Cornelius",
  "has_account": true,
  "has_avatar?": false,
  "id": 5625411010625,
  "last_name": "Potionmaker",
  "last_order": {},
  "name": "Cornelius Potionmaker",
  "orders": [],
  "orders_count": 1,
  "payment_methods": [],
  "phone": "+441314960905",
  "store_credit_account": {},
  "tags": [
    "newsletter"
  ],
  "tax_exempt": false,
  "total_spent": "56.00"
}

Outside of the above contexts, if the customer isn't logged into their account, the customer object returns nil


The frontend part layout/theme.liquid

    <div id="ai-chat-widget" style="position: fixed; bottom: 20px; right: 20px; z-index: 9999;">
        <button id="chat-icon" style="background: #008060; border: none; border-radius: 50%; width: 60px; height: 60px; cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.15); display: flex; align-items: center; justify-content: center;">
            <svg width="30" height="30" viewBox="0 0 24 24" fill="white"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/></svg>
        </button>

        <div id="chat-window" style="display: none; width: 350px; height: 450px; background: white; border-radius: 10px; box-shadow: 0 5px 20px rgba(0,0,0,0.2); flex-direction: column; overflow: hidden; margin-bottom: 10px;">
            <div style="background: #008060; color: white; padding: 15px; font-weight: bold;">Shop Assistant</div>
            <div id="chat-messages" style="flex: 1; padding: 10px; overflow-y: auto; font-size: 14px;"></div>
            <div style="padding: 10px; border-top: 1px solid #eee; display: flex;">
                <input type="text" id="chat-input" placeholder="Ask anything..." style="flex: 1; border: 1px solid #ddd; padding: 8px; border-radius: 4px; outline: none;">
                <button id="send-btn" style="background: #008060; color: white; border: none; margin-left: 5px; padding: 8px 15px; border-radius: 4px; cursor: pointer;">Send</button>
            </div>
        </div>
    </div>

    <script>
        // 1. Identify User and Role via Shopify Liquid
        // If customer object exists, they are a 'customer', otherwise 'guest'
        let userRole = 'customer'; 
        let userId = '{{ customer.id | default: "guest_session" }}';

        // Note: To identify an ADMIN, you might check a specific tag or 
        // use a different logic if they are logged into the Shopify admin bar.
        {% if customer.tags contains 'staff' %}
            userRole = 'admin';
        {% endif %}

        const chatIcon = document.getElementById('chat-icon');
        const chatWindow = document.getElementById('chat-window');
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        const chatMessages = document.getElementById('chat-messages');

        // 2. Pass role and id in the Connection URL
        const socket = new WebSocket(`ws://localhost:8000/ws/chat?role=${userRole}&user_id=${userId}`);

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            addMessage(data.reply, 'bot');
        };

        function addMessage(text, sender) {
            const div = document.createElement('div');
            div.style.margin = '5px 0';
            div.style.textAlign = sender === 'user' ? 'right' : 'left';
            div.innerHTML = `<span style="background: ${sender === 'user' ? '#e1ffc7' : '#f0f0f0'}; padding: 8px; border-radius: 10px; display: inline-block;">${text}</span>`;
            chatMessages.appendChild(div);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        sendBtn.onclick = () => {
            const msg = chatInput.value;
            if(msg) {
                addMessage(msg, 'user');
                // We just send the message text; the backend already knows the ID/Role from the connection
                socket.send(JSON.stringify({ message: msg }));
                chatInput.value = '';
            }
        };

        chatIcon.onclick = () => {
            chatWindow.style.display = chatWindow.style.display === 'none' ? 'flex' : 'none';
        };
    </script>
"""