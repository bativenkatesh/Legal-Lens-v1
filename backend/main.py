from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ocr.router import router as ocr_router
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uvicorn
import traceback
from db.mongo import connect_to_mongo, close_mongo_connection
from db.indexes import create_indexes
from db.conversation import get_messages, save_message
from utils.financial_context import get_user_financial_context
from utils.query_router import route_query
import uuid
# Import the new Orchestrator
try:
    from agent_core import orchestrator
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import agent_core: {e}")
    AGENT_AVAILABLE = False

app = FastAPI(title="Tax RAG Chatbot API (Agentic)")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)
app.include_router(ocr_router)


class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    # Made these optional because standard LangChain agents 
    # return a single string output by default.
    relevant_sections: Optional[List[dict]] = []
    relevant_articles: Optional[List[dict]] = []
    chat_history: Optional[List[dict]] = []
    debug_info: Optional[dict] = None
    status: str

@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()
    await create_indexes()
    """Check connections on startup"""
    if not AGENT_AVAILABLE:
        print("CRITICAL: Agent Core not loaded.")
    else:
        print("Backend ready with Agentic RAG Orchestrator!")
    print("MongoDB connected")


@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()
    print("MongoDB connection closed")

@app.get("/")
async def root():
    return {
        "message": "Tax RAG Chatbot API (Agentic Version)",
        "status": "running",
        "mode": "Agentic (Atlas + Local Mongo)"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "agent_loaded": AGENT_AVAILABLE
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Main chat endpoint delegating to the Agent Orchestrator"""
    if not AGENT_AVAILABLE:
        raise HTTPException(status_code=500, detail="Agent system not available")

    try:
        print(f"Received Query: {message.message}")
        user_id = "user_123" # Mock user_id
        conversation_id = message.conversation_id or "default_session"
        
        # 1. Retrieve existing chat history (per-message docs)
        history_docs = get_messages(user_id, conversation_id, limit=10)
        history_msgs = []
        raw_history = []
        
        from langchain_core.messages import HumanMessage, AIMessage
        for m in history_docs:
            # For the agent's internal list
            if m["role"] == "user":
                history_msgs.append(HumanMessage(content=m["content"]))
            else:
                history_msgs.append(AIMessage(content=m["content"]))
            # For the response back to frontend
            raw_history.append({"role": m["role"], "content": m["content"]})

        # 2. Save current user message to DB immediately
        save_message(user_id, conversation_id, "user", message.message)
        raw_history.append({"role": "user", "content": message.message})

        # 3. Detect Intent
        intent = await route_query(message.message)
        print("Detected Intent:", intent)

        if intent == "FINANCIAL":
            financial_context = await get_user_financial_context(user_id)
            input_text = f"User's financial summary: {financial_context}\n\nQuestion: {message.message}"
        else:
            input_text = message.message

        # 4. Invoke Orchestrator with chat_history
        result = orchestrator.invoke({
            "input": input_text,
            "chat_history": history_msgs
        })
        
        final_answer = result.get("output", "No response generated.")
        
        # 5. Save Assistant Response to DB
        save_message(user_id, conversation_id, "assistant", final_answer)
        raw_history.append({"role": "assistant", "content": final_answer})
        
        return ChatResponse(
            response=final_answer,
            relevant_sections=[], 
            relevant_articles=[], 
            chat_history=raw_history,
            debug_info={"full_result_keys": list(result.keys())},
            status="success"
        )
    
    except Exception as e:
        # Print the full error to your console so you can see what went wrong
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)