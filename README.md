# Agent Workflow Engine

A lightweight, Python-based workflow engine that enables users to define, execute, and monitor computational workflows through a graph-based approach. The system provides REST APIs for workflow management, real-time monitoring via WebSockets, and demonstrates its capabilities through practical workflow implementations.

## Features

- **Graph-based Workflow Definition**: Create workflows using nodes (functions) and edges (execution flow)
- **Tool Registry**: Register and manage reusable Python functions
- **State Management**: Persistent state handling with SQLite storage
- **Conditional Branching**: Support for if/else logic in workflows
- **Concurrent Execution**: Run multiple workflows simultaneously with proper isolation
- **REST API**: Complete HTTP API for workflow management
- **WebSocket Monitoring**: Real-time execution monitoring and event streaming
- **Error Handling**: Comprehensive error recovery and logging

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agent-workflow-engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the comprehensive example:
```bash
python workflow_example.py
```

4. Start the API server:
```bash
python -m app.main
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Web Layer                        │
├─────────────────────────────────────────────────────────────┤
│                  Workflow Engine Core                       │
├─────────────────────────────────────────────────────────────┤
│     Graph Manager    │    Execution Engine    │   Tool Registry │
├─────────────────────────────────────────────────────────────┤
│              State Management & Persistence                  │
├─────────────────────────────────────────────────────────────┤
│                   Storage Layer (SQLite)                    │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### Graph Manager
Handles workflow graph creation, validation, and storage. Ensures graphs are properly structured with valid node references and execution paths.

### Execution Engine
Executes workflow nodes, manages state transitions, handles control flow (branching and loops), and supports concurrent execution with proper isolation.

### Tool Registry
Manages reusable Python functions that can be called by workflow nodes. Provides registration, retrieval, and conflict prevention.

### State Manager
Handles state persistence, isolation between workflow runs, and consistency across the execution lifecycle.

## Complete Example

The `workflow_example.py` file demonstrates all system capabilities:

- **Mathematical Processing Workflow**: Shows basic operations with conditional branching
- **Code Review Workflow**: Analyzes Python code quality and generates reports
- **Tool Registration**: How to register custom functions
- **State Management**: How state flows between workflow nodes
- **Error Handling**: Graceful handling of execution failures

Run the example to see everything in action:
```bash
python workflow_example.py
```

## API Usage

### Create and Execute Workflows

```python
# 1. Register tools
tool_registry.register_tool("my_function", my_function, "Description")

# 2. Create workflow graph
workflow = GraphDefinition(
    name="My Workflow",
    nodes=[NodeDefinition(id="step1", function_name="my_function")],
    edges=[],
    entry_point="step1"
)

# 3. Execute workflow
graph_id = graph_manager.create_graph(workflow)
run_id = execution_engine.execute_workflow(graph_id, {"initial": "data"})

# 4. Monitor progress
status = execution_engine.get_execution_status(run_id)
```

### REST API Endpoints

- `POST /graph/create` - Create workflow graph
- `POST /graph/run` - Execute workflow
- `GET /graph/state/{run_id}` - Get execution status
- `GET /graph/logs/{run_id}` - Get execution logs
- `WebSocket /ws` - Real-time monitoring

## Project Structure

```
agent-workflow-engine/
├── app/
│   ├── api/           # REST API endpoints
│   ├── core/          # Core workflow engine components
│   ├── models/        # Pydantic data models
│   ├── storage/       # Database models and operations
│   └── tools/         # Built-in workflow tools
├── workflow_example.py # Comprehensive example
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## Configuration

Configure via environment variables or `AppConfig`:

```python
from app.config import AppConfig

config = AppConfig(
    app_name="My Workflow Engine",
    debug=True,
    database_url="sqlite:///workflows.db"
)
```

## License

This project is licensed under the MIT License.