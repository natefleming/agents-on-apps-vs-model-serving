"""Shared configuration for the LangGraph agent demo.

Central place for constants used across both deployment paths
(Model Serving and Databricks Apps).
"""

# Foundation Model API endpoint name
LLM_ENDPOINT: str = "databricks-claude-haiku-4-5"

# Unity Catalog coordinates for model registration
UC_CATALOG: str = "main"
UC_SCHEMA: str = "nfleming_agent_on_apps"
UC_MODEL_NAME: str = "demo_langgraph_agent"
UC_MODEL_FULL_NAME: str = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"

# System prompt for the agent
SYSTEM_PROMPT: str = """You are a helpful product assistant for an online retail store.
You can look up product information and perform calculations to help customers.
Always be concise and helpful. When using tools, explain what you found."""

# MLflow experiment path (used by Databricks Apps deployment)
MLFLOW_EXPERIMENT_PATH: str = "/Workspace/Shared/demo-agent-app"

# Databricks App name
APP_NAME: str = "demo-agent-app"
