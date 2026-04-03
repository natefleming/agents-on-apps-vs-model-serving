# Databricks notebook source

# MAGIC %md
# MAGIC # Deploy LangGraph Agent as a Databricks App
# MAGIC
# MAGIC This notebook demonstrates how to deploy the same LangGraph agent
# MAGIC as a Databricks App using **`mlflow.genai.agent_server.AgentServer`**.
# MAGIC
# MAGIC **Interface**: `AgentServer` with `@invoke` and `@stream` decorators
# MAGIC provides the same `/invocations` endpoint as Model Serving, with
# MAGIC streaming support via `{"stream": true}` in the request body.
# MAGIC
# MAGIC **Authentication**: Uses On-Behalf-Of (OBO) tokens -- the app runs
# MAGIC with the calling user's identity, inheriting their permissions.
# MAGIC
# MAGIC **MLflow Tracing**: Must be configured explicitly by setting the
# MAGIC tracking URI and experiment in the app code.
# MAGIC
# MAGIC **Key Advantage**: Full control over the application while still
# MAGIC using MLflow's standard serving infrastructure.

# COMMAND ----------

# MAGIC %pip install databricks-sdk>=0.102.0 databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
import os

# Add bundle root to sys.path and chdir so relative paths (app/, src/) resolve correctly.
# When run via DABs, notebook is at .bundle/<name>/<target>/files/notebooks/
# and app/ is at .bundle/<name>/<target>/files/app/
_notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_bundle_root = "/Workspace" + str(os.path.dirname(os.path.dirname(_notebook_path)))
sys.path.insert(0, _bundle_root)
os.chdir(_bundle_root)
print(f"Bundle root: {_bundle_root}")
print(f"CWD: {os.getcwd()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Review the app source code
# MAGIC
# MAGIC The app uses `mlflow.genai.agent_server.AgentServer` instead of
# MAGIC a custom FastAPI wrapper. Key files in `app/`:
# MAGIC
# MAGIC - `app.yaml` -- runs `python main.py` (AgentServer handles uvicorn)
# MAGIC - `main.py` -- agent definition + `@invoke`/`@stream` handlers
# MAGIC - `requirements.txt` -- Python dependencies
# MAGIC
# MAGIC AgentServer automatically provides:
# MAGIC - `POST /invocations` -- handles both sync and streaming requests
# MAGIC - `GET /health` -- health check endpoint
# MAGIC - Responses API format compatibility (same as Model Serving)

# COMMAND ----------

# Display the app configuration
with open("app/app.yaml") as f:
    print("=== app.yaml ===")
    print(f.read())

print()

with open("app/requirements.txt") as f:
    print("=== requirements.txt ===")
    print(f.read())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Deploy the app using Databricks CLI
# MAGIC
# MAGIC We use the Databricks CLI with the DEFAULT profile to create and deploy.
# MAGIC The CLI handles:
# MAGIC - Creating the app registration
# MAGIC - Uploading source code
# MAGIC - Building the container
# MAGIC - Starting the app

# COMMAND ----------

import subprocess
import json

from src.config import APP_NAME


def run_cli(cmd: str) -> str:
    """Run a Databricks CLI command and return output."""
    result = subprocess.run(
        cmd.split(),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
    return result.stdout


# Check if app already exists
existing_apps = run_cli("databricks apps list --profile DEFAULT --output json")
app_exists = APP_NAME in existing_apps

if not app_exists:
    print(f"Creating app '{APP_NAME}'...")
    create_output = run_cli(
        f"databricks apps create {APP_NAME} "
        f"--description 'LangGraph Agent Demo - AgentServer on Databricks Apps' "
        f"--profile DEFAULT"
    )
    print(create_output)
else:
    print(f"App '{APP_NAME}' already exists, will update.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Deploy the app source code
# MAGIC
# MAGIC Upload the `app/` directory as the source for the Databricks App.
# MAGIC The app runtime installs `requirements.txt` and runs the command
# MAGIC from `app.yaml` (`python main.py`), which starts AgentServer.

# COMMAND ----------

print(f"Deploying app '{APP_NAME}'...")
deploy_output = run_cli(
    f"databricks apps deploy {APP_NAME} "
    f"--source-code-path app/ "
    f"--profile DEFAULT"
)
print(deploy_output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Configure app resources
# MAGIC
# MAGIC Add the Foundation Model API endpoint as a **resource** so the
# MAGIC app's service principal has permission to query the LLM.
# MAGIC
# MAGIC **Via UI:** Compute > Apps > demo-agent-app > Resources > Add Serving Endpoint
# MAGIC
# MAGIC **Via SDK:**

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import AppResource, AppResourceServingEndpoint

w = WorkspaceClient(profile="DEFAULT")

try:
    w.apps.update(
        name=APP_NAME,
        resources=[
            AppResource(
                name="llm-endpoint",
                serving_endpoint=AppResourceServingEndpoint(
                    name="databricks-claude-sonnet-4-6",
                    permission="CAN_QUERY",
                ),
            )
        ],
    )
    print("Resource added successfully.")
except Exception as e:
    print(f"Note: You may need to add resources via the UI. Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Verify the app is running

# COMMAND ----------

app_info = run_cli(f"databricks apps get {APP_NAME} --profile DEFAULT --output json")
try:
    app_data = json.loads(app_info)
    print(f"App Name: {app_data.get('name')}")
    print(f"Status:   {app_data.get('status', {}).get('state', 'UNKNOWN')}")
    print(f"URL:      {app_data.get('url', 'Pending...')}")
except json.JSONDecodeError:
    print(app_info)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Test the deployed app (non-streaming)
# MAGIC
# MAGIC AgentServer exposes `/invocations` with the same Responses API
# MAGIC format as Model Serving -- the request/response shapes are identical.

# COMMAND ----------

import requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient(profile="DEFAULT")

app = w.apps.get(APP_NAME)
app_url = app.url

if app_url:
    headers = {
        "Authorization": f"Bearer {w.config.token}",
        "Content-Type": "application/json",
    }

    # Non-streaming request (same format as Model Serving)
    response = requests.post(
        f"{app_url}/invocations",
        headers=headers,
        json={
            "input": [
                {"role": "user", "content": "What's the price of a laptop?"}
            ]
        },
        timeout=60,
    )

    print("App response (non-streaming):")
    print(json.dumps(response.json(), indent=2))
else:
    print("App URL not yet available. Check deployment status.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Test the deployed app (streaming)
# MAGIC
# MAGIC Pass `"stream": true` in the request body to get Server-Sent Events.
# MAGIC This is the exact same interface as Model Serving streaming.

# COMMAND ----------

if app_url:
    headers = {
        "Authorization": f"Bearer {w.config.token}",
        "Content-Type": "application/json",
    }

    payload = {
        "input": [
            {"role": "user", "content": "Look up headphones and calculate 3 of them with 8% tax"}
        ],
        "stream": True,
    }

    print("App response (streaming):")
    with requests.post(
        f"{app_url}/invocations",
        headers=headers,
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        for line in resp.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data = decoded[6:]
                    if data != "[DONE]":
                        print(f"  {data}")
else:
    print("App URL not yet available.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary: Databricks Apps Deployment
# MAGIC
# MAGIC | Aspect | Detail |
# MAGIC |--------|--------|
# MAGIC | **Interface** | `AgentServer` with `@invoke` / `@stream` decorators |
# MAGIC | **Endpoint** | `POST /invocations` (same as Model Serving) |
# MAGIC | **Streaming** | `{"stream": true}` in request body |
# MAGIC | **Auth** | On-Behalf-Of (OBO) tokens -- runs as the calling user |
# MAGIC | **MLflow Tracing** | Manual config (tracking URI + experiment + autolog) |
# MAGIC | **Scaling** | Single instance (horizontal scaling coming) |
# MAGIC | **Deploy Time** | ~5 min |
# MAGIC | **Best For** | Custom apps, internal tools, rapid iteration |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparison: Model Serving vs Databricks Apps
# MAGIC
# MAGIC | Feature | Model Serving | Databricks Apps (AgentServer) |
# MAGIC |---------|--------------|-------------------------------|
# MAGIC | **Agent Interface** | `ResponsesAgent` | `@invoke` / `@stream` decorators |
# MAGIC | **API Format** | Responses API | Responses API (identical) |
# MAGIC | **Streaming** | `predict_stream` | `@stream` decorator |
# MAGIC | **Endpoint** | `/serving-endpoints/.../invocations` | `/invocations` |
# MAGIC | **Authentication** | Service Principal (auto) | OBO / App SP |
# MAGIC | **User Identity** | No (SP identity) | Yes (user's identity) |
# MAGIC | **MLflow Tracing** | Automatic | Manual setup |
# MAGIC | **Auto-scaling** | Yes (scale-to-zero) | No (single instance) |
# MAGIC | **Custom Endpoints** | No (chat API only) | Yes (extend FastAPI app) |
# MAGIC | **Deploy Complexity** | Low (1 function call) | Medium (app config) |
# MAGIC | **Iteration Speed** | Slow (~15 min) | Fast (~5 min) |
# MAGIC | **Cost Model** | Pay-per-token + compute | Fixed compute |
