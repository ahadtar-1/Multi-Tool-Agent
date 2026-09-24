"""
The module comprises of the User Interface for the Multi-tool Agent.
"""

import os
import asyncio
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gradio as gr
from agent.agent_with_multiple_tools_opt import build_agent, run_agent_safely

gemini_agent = build_agent()


def handle_question(query: str)-> str:


    if not query.strip():
        return "Please enter a valid question"
    
    async def run_with_timeout():
        agent_timeout_seconds = 30.0
        try:
            return await asyncio.wait_for(
                run_agent_safely(gemini_agent, query),
                timeout = agent_timeout_seconds
            )
        except asyncio.TimeoutError:
            return {"status": "system_error", "error": "The request timed out. Please try again."}

    response_dict = asyncio.run(run_with_timeout())
    if isinstance(response_dict, dict):
        if response_dict.get("status") == "success":
            return response_dict.get("data", "")
        else:
            return f"Error ({response_dict.get('error', 'Unknown error')})"
    
    return str(response_dict)


with gr.Blocks(theme=gr.themes.Glass(primary_hue="slate")) as demo:
    
    gr.Markdown("<h1 style='text-align: center;'>Multi-tool Agent</h1>")

    with gr.Row():
        question_box = gr.Textbox(label="Question", placeholder="Enter your question here...", lines = 4)
    
    send_button = gr.Button("Send")
    agent_output = gr.Textbox(label="Answer", visible=True, lines=6)

    send_button.click(
        handle_question,
        inputs = [question_box],
        outputs = agent_output
        )
