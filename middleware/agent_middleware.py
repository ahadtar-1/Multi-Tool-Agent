"""
The module comprises of the middleware for the Multi-tool Agent.
"""


import re
from typing import Any, Dict
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage


class SecurityViolationError(Exception):
    """
    Security Violation Exception for prompt injection
    """


    pass


class AgentInputSecurityMiddleware(AgentMiddleware):
    """
    Input guardrail for the Agent in order to check prompt injection
    """

    
    injection_patterns = [
        r"ignore previous instructions",
        r"disregard all prior rules",
        r"system prompt:"
    ]

    def before_model(self, state: Dict[str, Any], runtime: Any)-> Dict[str, Any]:
        msgs = state.get("messages", [])
        if msgs:
            last_msg_text = str(msgs[-1].content)
            for pattern in self.injection_patterns:
                if re.search(pattern, last_msg_text, re.IGNORECASE):
                    raise SecurityViolationError("A security violation has been detected: Input blocked by Agent Guardrail. Please check your query.")
        return state


class VerbatimOutputProtocolMiddleware(AgentMiddleware):
    """
    Output guardrail for the Agent. Enforces strict pass-through for the knowledge retriever tool.
    """
    
    
    def before_model(self, state: Dict[str, Any], runtime: Any)-> Dict[str, Any]:
        msgs = state.get("messages", [])
        if not msgs:
            return state

        current_tool_msgs = []
        for msg in reversed(msgs):
            if hasattr(msg, "type") and msg.type == "human":
                break
            if isinstance(msg, ToolMessage):
                current_tool_msgs.append(msg)

        if current_tool_msgs:
            tool_names = [getattr(m, "name", None) for m in current_tool_msgs]

            if "knowledge_retriever" in tool_names:
                retriever_content = next(m.content for m in current_tool_msgs if m.name == "knowledge_retriever")
                constraint_msg = SystemMessage(
                    content=(
                        f"PROTOCOL OVERRIDING REQUIRED: "
                        f"The following text retrieved from the knowledge_retriever MUST be included in your final answer EXACTLY word-by-word, character-for-character, without summarizing or changing a single word (unless not required by the query):\n\n"
                        f'"{retriever_content}"\n\n'
                        f"If other tools (like calculator or web_search) were called, you may synthesize and append their results, but you must not alter the text above (unless not required by the query)."
                    )
                )
                state["messages"].append(constraint_msg)

        return state
