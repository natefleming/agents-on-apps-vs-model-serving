"""Databricks App entry point using MLflow AgentServer.

This module deploys the same LangGraph agent as a Databricks App
using mlflow.genai.agent_server instead of a custom FastAPI wrapper.

Key differences from Model Serving deployment:

Authentication (OBO):
- Databricks Apps inject an X-Forwarded-Access-Token header on each request
  containing the calling user's OAuth token
- AgentServer's get_request_headers() captures these headers
- The token is used to create a WorkspaceClient acting as the user
- The app must declare user_api_scopes (e.g. "sql", "serving.serving-endpoints")
  so the platform knows which OAuth scopes to request from the user
- Users must authorize the app on first visit (OAuth consent screen)
- System-level resources (not user-specific) use the App Service Principal

MLflow Tracing:
- Must be configured explicitly (tracking URI + experiment)

AgentServer handles:
- The FastAPI app creation and configuration
- The /invocations endpoint (POST) for both sync and streaming
- Responses API format (ResponsesAgentRequest / ResponsesAgentResponse)
- Streaming via `"stream": true` in the request body
"""

from __future__ import annotations

import os
from typing import Generator

import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from mlflow.genai.agent_server import AgentServer, get_request_headers, invoke, stream
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Configuration from environment variables (set in app.yaml)
# ---------------------------------------------------------------------------
LLM_ENDPOINT: str = os.environ.get(
    "LLM_ENDPOINT", "databricks-claude-haiku-4-5"
)
MLFLOW_EXPERIMENT: str = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "/Workspace/Shared/demo-agent-app"
)

SYSTEM_PROMPT: str = """You are a helpful product assistant for an online retail store.
You can look up product information and perform calculations to help customers.
Always be concise and helpful. When using tools, explain what you found."""

# ---------------------------------------------------------------------------
# MLflow tracing setup (REQUIRED for Databricks Apps -- not automatic)
#
# Unlike Model Serving where tracing is auto-configured, Databricks Apps
# require explicit setup before any agent invocations:
#   1. set_tracking_uri("databricks") -> use workspace MLflow
#   2. set_experiment(...) -> where traces are stored
#   3. langchain.autolog() -> auto-instrument LangGraph spans
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(MLFLOW_EXPERIMENT)
mlflow.langchain.autolog()

# ---------------------------------------------------------------------------
# Tools (inlined for self-contained app deployment)
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

    Args:
        product_name: The name or keyword of the product to look up.
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
    return f"Product '{product_name}' not found. Available: {available}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression like '1299.99 * 0.9'.
    """
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters in expression."
    try:
        result = eval(expression)  # noqa: S307
        return (
            f"{expression} = {result:.4f}"
            if isinstance(result, float)
            else f"{expression} = {result}"
        )
    except (SyntaxError, ZeroDivisionError, TypeError) as e:
        return f"Error: {e}"


ALL_TOOLS: list = [lookup_product, calculator]


# ---------------------------------------------------------------------------
# LangGraph agent (same architecture as src/agent.py)
# ---------------------------------------------------------------------------
def _create_llm() -> ChatDatabricks:
    """Create the ChatDatabricks LLM with tools bound."""
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
    return llm.bind_tools(ALL_TOOLS)


def agent_node(state: ChatAgentState) -> dict[str, list]:
    """Invoke the LLM with the current messages."""
    llm_with_tools = _create_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response: AIMessage = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: ChatAgentState) -> str:
    """Route to tools or end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_graph() -> CompiledStateGraph:
    """Build and compile the LangGraph agent."""
    graph = StateGraph(ChatAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ChatAgentToolNode(ALL_TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


# Build the graph once at module load
_graph: CompiledStateGraph = build_graph()


# ---------------------------------------------------------------------------
# OBO header capture
#
# Databricks Apps inject X-Forwarded-Access-Token and X-Forwarded-User
# headers on each request. get_request_headers() from AgentServer captures
# them. These can be used to create a WorkspaceClient acting as the user:
#
#   headers = get_request_headers()
#   token = headers.get("x-forwarded-access-token")
#   user = headers.get("x-forwarded-user")
#   w = WorkspaceClient(token=token, auth_type="pat")
#
# This enables per-user access control: the agent can query data,
# call endpoints, or access resources as the calling user -- not the
# app's service principal.
# ---------------------------------------------------------------------------
def _log_obo_user() -> None:
    """Log the OBO user from forwarded headers (if present)."""
    headers: dict[str, str] = get_request_headers()
    if headers:
        user = headers.get("x-forwarded-user") or headers.get(
            "X-Forwarded-User", "unknown"
        )
        has_token = bool(
            headers.get("x-forwarded-access-token")
            or headers.get("X-Forwarded-Access-Token")
        )
        print(f"OBO request from user={user}, has_token={has_token}")


# ---------------------------------------------------------------------------
# AgentServer handler functions
#
# The @invoke decorator registers the non-streaming handler.
# The @stream decorator registers the streaming handler.
# AgentServer exposes both via POST /invocations:
#   - Non-streaming: POST /invocations with {"input": [...]}
#   - Streaming:     POST /invocations with {"input": [...], "stream": true}
# ---------------------------------------------------------------------------
@invoke()
def handle_invoke(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Handle non-streaming agent invocations.

    Captures OBO headers from the request, then collects all completed
    output items from the streaming pipeline.

    Args:
        request: Incoming request in Responses API format.

    Returns:
        ResponsesAgentResponse with all output items.
    """
    _log_obo_user()
    outputs = [
        event.item
        for event in _stream_agent(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(
        output=outputs,
        custom_outputs=request.custom_inputs,
    )


@stream()
def handle_stream(
    request: ResponsesAgentRequest,
) -> Generator[ResponsesAgentStreamEvent, None, None]:
    """Handle streaming agent invocations.

    Captures OBO headers, then streams ResponsesAgentStreamEvents
    as the LangGraph agent processes the request.

    Args:
        request: Incoming request in Responses API format.

    Yields:
        ResponsesAgentStreamEvent for each output chunk.
    """
    _log_obo_user()
    yield from _stream_agent(request)


def _stream_agent(
    request: ResponsesAgentRequest,
) -> Generator[ResponsesAgentStreamEvent, None, None]:
    """Core streaming logic shared by invoke and stream handlers.

    Converts Responses API input to chat-completions format,
    runs the LangGraph agent, and yields Responses API events.

    Args:
        request: Incoming request in Responses API format.

    Yields:
        ResponsesAgentStreamEvent for each output message.
    """
    cc_msgs = to_chat_completions_input(
        [item.model_dump() for item in request.input]
    )

    for update in _graph.stream(
        {"messages": cc_msgs}, stream_mode=["updates"]
    ):
        mode_name, events = update
        for node_name, node_data in events.items():
            msgs = node_data.get("messages", []) if isinstance(node_data, dict) else []
            # ChatAgentState may return messages as dicts or AIMessage objects.
            # Convert dicts to AIMessage so output_to_responses_items_stream works.
            converted = []
            for msg in msgs:
                if isinstance(msg, dict):
                    converted.append(AIMessage(content=msg.get("content", ""), additional_kwargs=msg.get("additional_kwargs", {})))
                else:
                    converted.append(msg)
            yield from output_to_responses_items_stream(converted)


# ---------------------------------------------------------------------------
# AgentServer setup
#
# AgentServer creates a FastAPI app with:
#   - POST /invocations  (handles both sync and streaming)
#   - GET  /health
#
# For Databricks Apps, app.yaml points uvicorn at "main:app"
# ---------------------------------------------------------------------------
agent_server = AgentServer("ResponsesAgent")
app = agent_server.app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
