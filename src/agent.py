"""LangGraph agent with ResponsesAgent interface for Databricks deployment.

This module defines a ReAct-style LangGraph agent that:
- Uses Databricks Foundation Model API as the LLM
- Binds product lookup and calculator tools
- Implements mlflow.pyfunc.ResponsesAgent for Model Serving compatibility
- Supports both streaming and non-streaming inference via Responses API format
- Enables MLflow tracing via langchain autolog

This file is SELF-CONTAINED for models-from-code logging. All config and
tools are inlined so the model artifact has no external dependencies on
sibling modules. This is required because Model Serving loads only this
file in an isolated container.

Both deployment paths (Model Serving and Databricks Apps via AgentServer)
use the same agent architecture -- only the infrastructure wrapper differs.
"""

from __future__ import annotations

from typing import Generator

import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

# ---------------------------------------------------------------------------
# Configuration (inlined for self-contained model artifact)
# ---------------------------------------------------------------------------
LLM_ENDPOINT: str = "databricks-claude-haiku-4-5"

SYSTEM_PROMPT: str = """You are a helpful product assistant for an online retail store.
You can look up product information and perform calculations to help customers.
Always be concise and helpful. When using tools, explain what you found."""

# ---------------------------------------------------------------------------
# Enable MLflow LangChain/LangGraph auto-tracing
# Only enable at serving time, not during model logging (which executes
# this file for validation and can trigger rate-limited LLM calls).
# Model Serving sets ENABLE_MLFLOW_TRACING=true in the environment.
# ---------------------------------------------------------------------------
import os as _os
if _os.environ.get("ENABLE_MLFLOW_TRACING", "").lower() == "true" or _os.environ.get("IS_SERVING_REQUEST"):
    mlflow.langchain.autolog()

# ---------------------------------------------------------------------------
# Tools (inlined for self-contained model artifact)
# ---------------------------------------------------------------------------
_PRODUCT_CATALOG: dict[str, dict[str, str | float | bool]] = {
    "laptop": {
        "name": "ProBook Laptop 15",
        "price": 1299.99,
        "in_stock": True,
        "category": "Electronics",
        "description": "15-inch display, 16GB RAM, 512GB SSD",
    },
    "headphones": {
        "name": "SoundMax Pro Headphones",
        "price": 249.99,
        "in_stock": True,
        "category": "Electronics",
        "description": "Noise-cancelling, wireless, 30hr battery",
    },
    "backpack": {
        "name": "TravelPro Backpack",
        "price": 89.99,
        "in_stock": False,
        "category": "Accessories",
        "description": "Water-resistant, laptop compartment, 35L",
    },
    "keyboard": {
        "name": "MechType Pro Keyboard",
        "price": 179.99,
        "in_stock": True,
        "category": "Electronics",
        "description": "Mechanical, RGB, wireless/wired, hot-swappable",
    },
    "monitor": {
        "name": "UltraView 27 Monitor",
        "price": 549.99,
        "in_stock": True,
        "category": "Electronics",
        "description": "27-inch 4K, USB-C, HDR400, adjustable stand",
    },
}


@tool
def lookup_product(product_name: str) -> str:
    """Look up product information by name.

    Use this tool when a customer asks about a specific product,
    its price, availability, or description.

    Args:
        product_name: The name or keyword of the product to look up.
                      Examples: 'laptop', 'headphones', 'keyboard'.
    """
    key = product_name.strip().lower()
    if key in _PRODUCT_CATALOG:
        product = _PRODUCT_CATALOG[key]
        stock_status = "In Stock" if product["in_stock"] else "Out of Stock"
        return (
            f"Product: {product['name']}\n"
            f"Price: ${product['price']:.2f}\n"
            f"Status: {stock_status}\n"
            f"Category: {product['category']}\n"
            f"Description: {product['description']}"
        )
    available = ", ".join(_PRODUCT_CATALOG.keys())
    return f"Product '{product_name}' not found. Available products: {available}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Use this tool for any calculations like totals, discounts,
    tax computations, or unit conversions.

    Args:
        expression: A mathematical expression to evaluate.
                    Examples: '1299.99 * 0.9', '249.99 + 179.99', '100 / 3'.
    """
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Expression contains invalid characters."
    try:
        result = eval(expression)  # noqa: S307
        return f"{expression} = {result:.4f}" if isinstance(result, float) else f"{expression} = {result}"
    except (SyntaxError, ZeroDivisionError, TypeError) as e:
        return f"Error evaluating '{expression}': {e}"


ALL_TOOLS: list = [lookup_product, calculator]


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------
def _create_llm() -> ChatDatabricks:
    """Create the ChatDatabricks LLM with tools bound."""
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
    return llm.bind_tools(ALL_TOOLS)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
def agent_node(state: ChatAgentState) -> dict[str, list]:
    """Invoke the LLM with the current message history."""
    llm_with_tools = _create_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response: AIMessage = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: ChatAgentState) -> str:
    """Route to tools if the last message has tool calls, else end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph() -> CompiledStateGraph:
    """Construct and compile the LangGraph ReAct agent graph."""
    tool_node = ChatAgentToolNode(ALL_TOOLS)
    graph = StateGraph(ChatAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ---------------------------------------------------------------------------
# ResponsesAgent wrapper
# ---------------------------------------------------------------------------
class LangGraphResponsesAgent(ResponsesAgent):
    """Wraps a LangGraph compiled graph as an MLflow ResponsesAgent."""

    def __init__(self) -> None:
        self.graph: CompiledStateGraph = build_graph()

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Run the agent to completion and return all response items."""
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(
            output=outputs,
            custom_outputs=request.custom_inputs,
        )

    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream agent responses as ResponsesAgentStreamEvents."""
        cc_msgs = to_chat_completions_input(
            [item.model_dump() for item in request.input]
        )
        for _, events in self.graph.stream(
            {"messages": cc_msgs}, stream_mode=["updates"]
        ):
            for node_data in events.values():
                msgs = node_data.get("messages", []) if isinstance(node_data, dict) else []
                # ChatAgentState may return messages as dicts or AIMessage objects.
                # Convert dicts to AIMessage so output_to_responses_items_stream works.
                converted = []
                for msg in msgs:
                    if isinstance(msg, dict):
                        converted.append(
                            AIMessage(
                                content=msg.get("content", ""),
                                additional_kwargs=msg.get("additional_kwargs", {}),
                            )
                        )
                    else:
                        converted.append(msg)
                yield from output_to_responses_items_stream(converted)


# ---------------------------------------------------------------------------
# Module-level agent instance + MLflow model export
# ---------------------------------------------------------------------------
agent = LangGraphResponsesAgent()
set_model(agent)
