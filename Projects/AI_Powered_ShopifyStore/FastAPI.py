"""
FastAPI server for stores frontend chatbot widget.

Usage:
    uvicorn FastAPI:app --host 0.0.0.0 --port 8000
"""

import os
import json
from graph import graph
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY_2")
os.environ["LANGCHAIN_PROJECT"] = "shopify-agent"

app = FastAPI()



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

 