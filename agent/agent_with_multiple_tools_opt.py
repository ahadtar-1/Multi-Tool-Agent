"""
The module contains the Multi-tool Agent.
"""

import os
import time
import asyncio
import json
import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv, find_dotenv
import langchain
from langchain.tools import ToolException
from langchain.agents import create_agent
from middleware.agent_middleware import SecurityViolationError, AgentInputSecurityMiddleware, VerbatimOutputProtocolMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai import types
from agent_tools.tools import calculator, knowledge_retriever, web_search

_ = load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GOOGLE_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langsmith_tracing_message = os.getenv("LANGSMITH_TRACING_V2")
langsmith_project = os.getenv("LANGSMITH_PROJECT")

gemini_threepointfive_flashlitellm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", 
    temperature=1.0,
    max_retries=2,
    model_kwargs={"generation_config": {"thinking_config": {"thinking_budget": 0}}}
)

middlewares = [
    AgentInputSecurityMiddleware(), 
    VerbatimOutputProtocolMiddleware()
]

tools = [calculator, knowledge_retriever, web_search]


def build_agent():
    """
    Builds and returns the Agent.    
    """


    return create_agent(
    gemini_threepointfive_flashlitellm,
    tools=tools,
    middleware=middlewares
    )

    
async def run_agent_safely(agent, query: str):
    """
    Executes the Agent.

    Parameters
    ----------
    agent
        The Agent

    query: str
        The input query, sent by the user, to the Agent

    Returns
    -------
    str
        The response of the Agent to the user's query
    
    """
    
    
    try:
        current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%B %d, %Y at %H:%M %Z")
        dynamic_system_prompt = (f"<role>\nYou are a professional assistant.\n</role>\n<instructions>\nBe helpful, polite, and direct.\nUse formal language only in your answers. Do not use any special characters.\n\nCurrent Date: {current_date}\n\nIMPORTANT NOTE: When calling the web_search tool is decided, use the Current Date only for time-sensitive queries unless another date is explicitly specified.\n</instructions>")
        result = await agent.ainvoke({"messages": [{"role": "system", "content": dynamic_system_prompt}, {"role": "user", "content": query}]}, config={"recursion_limit": 50})
        #content = ""
        #if isinstance(result, dict) and "messages" in result:
        #    last_msg = result["messages"][-1]
        #    if hasattr(last_msg, "content"):
        #        raw_content = last_msg.content
        #        content = raw_content[0].get("text", "") if isinstance(raw_content, list) else str(raw_content)
        #else:
        #    content = str(result)
        return result
        #return {"status": "success", "data": content}
    except SecurityViolationError as e:
        return {"status": "security_violation", "error": str(e)}
    except (ToolException, RuntimeError) as te:
        return {"status": "tool_error", "error": str(te)}
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            return {"status": "quota_exhausted", "error": "The AI model provider's quota has been exhausted. Please check your API key."}        
        return {"status": "system_error", "error": "Network timeout. Please try again."}

if __name__ == "__main__":
    gemini_agent = build_agent()
    query = "Look up the population of France and add 1,000,000 to it."
    print(asyncio.run(run_agent_safely(gemini_agent, query)))