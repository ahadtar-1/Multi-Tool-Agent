"""
The module comprises of the API Endpoints for FastAPI. 
"""

import os
import asyncio
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, status
from schemas.schemas import ChatRequest, ChatResponse
from frontend.gradio_frontend import demo
from agent.agent_with_multiple_tools_opt import build_agent, run_agent_safely

_ = load_dotenv(find_dotenv())

app = FastAPI()
agent_executor = build_agent()


@app.post("/agentchat", response_model=ChatResponse)
async def gemini_agent_endpoint(body: ChatRequest): 
    """
    The API endpoint for the Agent

    Parameters
    ----------
    body: ChatRequest
        The ChatRequest body containing the user's query

    Returns
    -------
    ChatResponse
        The ChatResponse body containing the Agent's Response

    """  
    
    
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")
    agent_timeout_seconds = 30.0
    try:    
        response_dict = await asyncio.wait_for(
            run_agent_safely(agent_executor, query),
            timeout = agent_timeout_seconds
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="The request timed out while waiting for the agent. Please try again."
        )
    status_code = response_dict.get("status")
    if status_code == "success":
        return ChatResponse(agent_response = response_dict["data"])
    elif status_code == "security_violation":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=response_dict["error"])
    elif status_code == "quota_exhausted":
        raise HTTPException(status_code=429, detail=response_dict["error"])
    elif status_code == "tool_error":
        raise HTTPException(status_code=502, detail=response_dict["error"])
    elif status_code == "system_error":
        raise HTTPException(status_code=500, detail=response_dict["error"])
    
    return ChatResponse(agent_response=str(response_dict.get("error", "Unknown error")))


app = gr.mount_gradio_app(
    app,
    demo,
    path="/"
)
