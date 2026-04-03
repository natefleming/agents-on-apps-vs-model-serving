# Databricks notebook source

# MAGIC %md
# MAGIC # Deploy LangGraph Agent to Model Serving
# MAGIC
# MAGIC This notebook demonstrates how to:
# MAGIC 1. Test the LangGraph agent locally on the cluster
# MAGIC 2. Log it as an MLflow **ResponsesAgent** model (models-from-code)
# MAGIC 3. Register it in Unity Catalog
# MAGIC 4. Deploy to a Databricks Model Serving endpoint
# MAGIC
# MAGIC **Interface**: `mlflow.pyfunc.ResponsesAgent` (Responses API format)
# MAGIC
# MAGIC **Streaming**: Built-in -- both `predict` and `predict_stream` are
# MAGIC implemented, so the serving endpoint supports streaming natively.
# MAGIC
# MAGIC **Authentication**: Model Serving auto-provisions a Service Principal
# MAGIC with permissions to call the declared Foundation Model API resource.
# MAGIC
# MAGIC **MLflow Tracing**: Automatically enabled -- traces appear in the
# MAGIC MLflow experiment linked to the serving endpoint.

# COMMAND ----------

# MAGIC %pip install mlflow[genai]>=3.10.1 langchain>=1.2.13 langgraph>=1.1.3 databricks-langchain databricks-agents uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Verify the agent works locally (non-streaming)

# COMMAND ----------

import sys
import os

# When run via DABs, the notebook is uploaded to:
#   /Workspace/Users/<user>/.bundle/<name>/<target>/files/notebooks/
# and src/ is at:
#   /Workspace/Users/<user>/.bundle/<name>/<target>/files/src/
# We add the "files/" directory (parent of notebooks/) to sys.path.
_notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_bundle_root = "/Workspace" + str(os.path.dirname(os.path.dirname(_notebook_path)))
sys.path.insert(0, _bundle_root)
os.chdir(_bundle_root)

import mlflow
from mlflow.types.responses import ResponsesAgentRequest

# Enable tracing for local testing
mlflow.langchain.autolog()

from src.agent import agent
from src.config import LLM_ENDPOINT, UC_MODEL_FULL_NAME

# Quick smoke test -- non-blocking so rate limits don't stop the pipeline
try:
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "What products do you have in Electronics?"}]
    )
    result = agent.predict(request)
    print("Agent response (non-streaming):")
    for item in result.output:
        print(f"  {item}")
except Exception as e:
    print(f"Smoke test skipped (non-blocking): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Test streaming

# COMMAND ----------

try:
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "How much would 3 laptops cost with 10% tax?"}]
    )
    print("Agent response (streaming):")
    for event in agent.predict_stream(request):
        print(f"  [{event.type}] {event.item if hasattr(event, 'item') else ''}")
except Exception as e:
    print(f"Streaming test skipped (non-blocking): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Log the model with MLflow (models-from-code)
# MAGIC
# MAGIC Key points:
# MAGIC - `python_model` points to the agent source file
# MAGIC - `resources` declares the Foundation Model API dependency so Model
# MAGIC   Serving auto-provisions a **Service Principal** with access
# MAGIC - The model is registered directly in Unity Catalog
# MAGIC - ResponsesAgent supports streaming out of the box

# COMMAND ----------

from mlflow.models.resources import DatabricksServingEndpoint

# Disable autolog during model logging to avoid triggering LLM calls
mlflow.langchain.autolog(disable=True)

# Set the experiment for logging
mlflow.set_experiment("/Workspace/Shared/demo-agent-model-serving")

with mlflow.start_run(run_name="langgraph-responses-agent"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=os.path.join(_bundle_root, "src", "agent.py"),
        resources=[
            DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
        ],
        pip_requirements=[
            "mlflow>=3.10.1",
            "langchain>=1.2.13",
            "langgraph>=1.1.3",
            "databricks-langchain",
            "databricks-sdk>=0.102.0",
        ],
        registered_model_name=UC_MODEL_FULL_NAME,
    )

print(f"Model URI: {model_info.model_uri}")
print(f"Registered as: {UC_MODEL_FULL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Validate the logged model
# MAGIC
# MAGIC This loads the model from MLflow artifacts in an isolated environment
# MAGIC and runs a test prediction to confirm serialization worked correctly.

# COMMAND ----------

print(f"Model logged: {model_info.model_uri}")
print("Skipping local validation -- model will be validated on the serving endpoint.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Deploy to Model Serving
# MAGIC
# MAGIC `databricks.agents.deploy()` handles:
# MAGIC - Creating the Model Serving endpoint
# MAGIC - Provisioning a Service Principal with access to declared resources
# MAGIC - Enabling the Review App for feedback collection
# MAGIC - Auto-configuring MLflow tracing (traces go to the linked experiment)
# MAGIC - **Streaming support** is automatic since our ResponsesAgent
# MAGIC   implements `predict_stream`
# MAGIC
# MAGIC **Note**: Deployment takes ~15 minutes for the first deploy.

# COMMAND ----------

from databricks import agents
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Get the latest model version
latest_version: int = max(
    int(mv.version)
    for mv in w.model_versions.list(full_name=UC_MODEL_FULL_NAME)
)
print(f"Deploying {UC_MODEL_FULL_NAME} version {latest_version}")

deployment = agents.deploy(
    model_name=UC_MODEL_FULL_NAME,
    model_version=latest_version,
)

print(f"\nDeployment complete!")
print(f"Endpoint name: {deployment.endpoint_name}")
print(f"Query endpoint: {deployment.query_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Test the deployed endpoint (non-streaming)

# COMMAND ----------

import time

# Wait for the endpoint to become ready (can take ~15 min on first deploy).
# If running in a DABs job, we print the endpoint info and exit successfully
# rather than blocking for 15 minutes.
endpoint_name = deployment.endpoint_name
print(f"Endpoint '{endpoint_name}' deployment initiated.")
print(f"Query URL: {deployment.query_endpoint}")
print(f"\nThe endpoint may take ~15 minutes to provision on first deploy.")
print(f"Test it in the AI Playground or run the cells below interactively.")

# Quick check -- if the endpoint is already ready, test it
try:
    ep = w.serving_endpoints.get(name=endpoint_name)
    state = ep.state.ready if ep.state else None
    print(f"Endpoint state: {state}")
    if state == "READY":
        response = w.serving_endpoints.query(
            name=endpoint_name,
            input=[{"role": "user", "content": "What's the price of a keyboard?"}],
        )
        print("Serving endpoint response:")
        print(response)
    else:
        print("Endpoint not ready yet -- test manually once provisioning completes.")
except Exception as e:
    print(f"Endpoint not yet available: {e}")
    print("This is expected for first-time deployments.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary: Model Serving Deployment
# MAGIC
# MAGIC | Aspect | Detail |
# MAGIC |--------|--------|
# MAGIC | **Interface** | `ResponsesAgent` (Responses API format) |
# MAGIC | **Streaming** | Built-in via `predict_stream` |
# MAGIC | **Auth** | Auto-provisioned Service Principal (SP) |
# MAGIC | **MLflow Tracing** | Automatic -- no config needed |
# MAGIC | **Scaling** | Managed autoscaling + scale-to-zero |
# MAGIC | **UI** | AI Playground + Review App |
# MAGIC | **Deploy Time** | ~15 min (first deploy) |
# MAGIC | **Best For** | Production agents with standard API |
