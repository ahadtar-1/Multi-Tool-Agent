"""
The module comprises of the Pydantic models. 
"""

from typing import Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    agent_response: str
    