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

3. Initialize the database:
```bash
python app/startup.py db init
```

## Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
```

Key configuration options:
- `WORKFLOW_ENGINE_DATABASE_URL`: Database connection URL
- `WORKFLOW_ENGINE_MAX_CONCURRENT_EXECUTIONS`: Maximum concurrent workflows
- `WORKFLOW_ENGINE_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `WORKFLOW_ENGINE_HOST`: Server host (default: 0.0.0.0)
- `WORKFLOW_ENGINE_PORT`: Server port (default: 8000)

## Running the Application

### Using the startup script (recommended):
```bash
# Run with default settings
python app/startup.py run

# Run with custom configuration
python app/startup.py --host 127.0.0.1 --port 8080 --debug run

# Run in development mode
python app/startup.py --env development run

# Show configuration
python app/startup.py config show

# Run health checks
python app/startup.py health --detailed
```

### Using the main module directly:
```bash
python -m app.main
```

### Using Docker:
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t workflow-engine .
docker run -p 8000:8000 workflow-engine
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

## Health Checks

The application provides several health check endpoints:

- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed component health status
- `GET /health/ready` - Readiness check for container orchestration
- `GET /health/live` - Liveness check for container orchestration

## Database Management

```bash
# Initialize database tables
python app/startup.py db init

# Run database migrations
python app/startup.py db migrate

# Reset database (drop and recreate)
python app/startup.py db reset
```

## Development Status

This project implements a complete workflow engine with modern Python architecture and clean design principles.