# Agent Workflow Engine

A lightweight, Python-based workflow engine that enables users to define, execute, and monitor computational workflows through a graph-based approach.

## Features

- Graph-based workflow definition with nodes and edges
- REST API for workflow management
- Real-time monitoring via WebSockets
- State persistence and management
- Tool registry for reusable functions
- Concurrent execution support

## Project Structure

```
app/
├── api/           # FastAPI endpoints and WebSocket handlers
├── core/          # Core workflow engine components
├── models/        # Pydantic data models
├── storage/       # Database models and storage layer
└── main.py        # FastAPI application entry point
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python -m app.main
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

## Development Status

This project implements a complete workflow engine with modern Python architecture and clean design principles.