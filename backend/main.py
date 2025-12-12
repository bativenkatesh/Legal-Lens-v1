from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uvicorn
import traceback

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

class ChatMessage(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str
    # Made these optional because standard LangChain agents 
    # return a single string output by default.
    relevant_sections: Optional[List[dict]] = []
    relevant_articles: Optional[List[dict]] = []
    debug_info: Optional[dict] = None
    status: str

@app.on_event("startup")
async def startup_event():
    """Check connections on startup"""
    if not AGENT_AVAILABLE:
        print("CRITICAL: Agent Core not loaded.")
    else:
        print("Backend ready with Agentic RAG Orchestrator!")

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

        # 1. CRITICAL FIX: Use .invoke({"input": ...})
        # The orchestrator is a LangChain AgentExecutor
        result = orchestrator.invoke({"input": message.message})
        
        # 2. Extract the final text response
        # AgentExecutor returns a dict with 'input', 'output', and optionally 'intermediate_steps'
        final_answer = result.get("output", "No response generated.")
        
        # 3. Optional: Parse intermediate steps for debug info (advanced)
        # For now, we will return empty lists to avoid 500 errors
        # if the agent doesn't return structured data.
        
        return ChatResponse(
            response=final_answer,
            relevant_sections=[], # Placeholder: Standard agents don't return these cleanly split
            relevant_articles=[], # Placeholder
            debug_info={"full_result_keys": list(result.keys())},
            status="success"
        )
    
    except Exception as e:
        # Print the full error to your console so you can see what went wrong
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)