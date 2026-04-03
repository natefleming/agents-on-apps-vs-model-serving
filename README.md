# Agent on Apps: Model Serving vs Databricks Apps

> **[View the Presentation: Deploying AI Agents on Databricks](https://natefleming.github.io/agents-on-apps-vs-model-serving/presentation/model-serving-vs-apps.html)** — An interactive slide deck comparing Model Serving and Databricks Apps as deployment targets for LangGraph agents.

Demo project comparing two deployment paths for the same LangGraph agent on Databricks.

## Project Structure

```
agent-on-apps/
├── src/
│   ├── __init__.py
│   ├── config.py          # Shared constants (endpoints, UC coordinates)
│   ├── tools.py            # Product lookup + calculator tools
│   └── agent.py            # LangGraph agent + ChatAgent wrapper
├── app/
│   ├── app.yaml            # Databricks Apps configuration
│   ├── main.py             # FastAPI server (self-contained for app deploy)
│   └── requirements.txt    # App runtime dependencies
├── notebooks/
│   ├── 01_deploy_model_serving.py   # Log, register, deploy to serving
│   └── 02_deploy_databricks_app.py  # Deploy as Databricks App
└── pyproject.toml
```

## The Agent

A **LangGraph ReAct agent** that:
- Uses **ChatDatabricks** (Llama 3.3 70B via Foundation Model API)
- Has two tools: product catalog lookup and math calculator
- Implements `mlflow.pyfunc.ChatAgent` for Model Serving compatibility
- Enables MLflow tracing via `mlflow.langchain.autolog()`

## Deployment Paths

### Path 1: Model Serving

```
Notebook 01 → MLflow log_model → Unity Catalog → agents.deploy() → Serving Endpoint
```

- **Auth**: Auto-provisioned Service Principal
- **Tracing**: Automatic (zero config)
- **Scaling**: Managed autoscaling + scale-to-zero
- **Interface**: OpenAI-compatible chat completions API

### Path 2: Databricks Apps

```
app/ directory → databricks apps deploy → FastAPI server → Custom endpoints
```

- **Auth**: On-Behalf-Of (OBO) tokens (user identity)
- **Tracing**: Manual setup (tracking URI + experiment + autolog)
- **Scaling**: Single instance (horizontal scaling coming)
- **Interface**: Custom REST API (you define the endpoints)

## Quick Start

### Prerequisites

- Databricks workspace with Foundation Model API enabled
- Databricks CLI configured with DEFAULT profile
- Unity Catalog with `main.agents` schema

### Install dependencies

```bash
uv sync
```

### Deploy to Model Serving

Run `notebooks/01_deploy_model_serving.py` on a Databricks cluster.

### Deploy as Databricks App

Run `notebooks/02_deploy_databricks_app.py` on a Databricks cluster,
or use the CLI directly:

```bash
databricks apps create demo-agent-app --profile DEFAULT
databricks apps deploy demo-agent-app --source-code-path app/ --profile DEFAULT
```

## Key Differences

| Feature | Model Serving | Databricks Apps |
|---------|--------------|-----------------|
| Auth | Service Principal (auto) | OBO (user identity) |
| MLflow Tracing | Automatic | Manual config |
| Auto-scaling | Yes | No (single instance) |
| Custom UI | No | Yes |
| Deploy Time | ~15 min | ~5 min |
| Cost Model | Pay-per-token | Fixed compute |

