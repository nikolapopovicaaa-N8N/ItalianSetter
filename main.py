import os
import logging
import uvicorn
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agent import graph

# --- Setup & Configuration ---
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LangGraph Agent API",
    description="Production-ready API for LangGraph conversational agent.",
    version="1.0.0"
)

# Add CORS Middleware (Allow all origins as requested)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class ConfigurableConfig(BaseModel):
    thread_id: str = Field(..., description="Unique identifier for the conversation thread")

class RequestConfig(BaseModel):
    configurable: ConfigurableConfig

class RequestInput(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="List of messages, e.g., [{'role': 'user', 'content': '...'}]")

class InvokeRequest(BaseModel):
    input: RequestInput
    config: RequestConfig

class InvokeResponseOutput(BaseModel):
    messages: List[Dict[str, Any]]

class InvokeResponse(BaseModel):
    output: Optional[InvokeResponseOutput] = None
    error: Optional[str] = None
    status: str = "success"

# --- Endpoints ---

@app.get("/")
async def health_check():
    """
    Health check endpoint.
    Returns: {"status": "ok"}
    """
    logger.info("Health check requested")
    return {"status": "ok"}

@app.post("/invoke", response_model=InvokeResponse)
async def invoke_agent(request: InvokeRequest):
    """
    Invokes the LangGraph agent with the provided input and configuration.
    """
    try:
        logger.info(f"Received invoke request for thread_id: {request.config.configurable.thread_id}")
        
        # Prepare graph input
        graph_input = {"messages": request.input.messages}
        
        # Prepare graph config
        graph_config = {"configurable": {"thread_id": request.config.configurable.thread_id}}
        
        # Invoke the graph
        final_state = graph.invoke(graph_input, config=graph_config)
        
        # Extract messages
        output_messages = final_state.get("messages", [])
        
        # Serialize messages for response
        serialized_msgs = []
        for msg in output_messages:
            if hasattr(msg, "model_dump"):
                serialized_msgs.append(msg.model_dump())
            elif hasattr(msg, "dict"):
                 serialized_msgs.append(msg.dict())
            else:
                 serialized_msgs.append(msg)
                 
        return InvokeResponse(
            output=InvokeResponseOutput(messages=serialized_msgs),
            status="success"
        )

    except Exception as e:
        logger.error(f"Error invoking agent: {str(e)}", exc_info=True)
        # Return error format as requested: {"error": "...", "status": "failed"}
        return InvokeResponse(
            output=None,
            error=str(e),
            status="failed"
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
