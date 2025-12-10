"""Main FastAPI application for the workflow engine."""

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.logging import setup_logging, get_logger
from app.storage.database import create_tables
from app.core.graph_manager import GraphManager
from app.core.execution_engine import ExecutionEngine
from app.core.tool_registry import ToolRegistry
from app.core.state_manager import StateManager
from app.core.websocket_manager import WebSocketManager
from app.api.endpoints import router, init_dependencies


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    setup_logging()
    logger = get_logger(__name__)
    logger.info("Starting Agent Workflow Engine")
    
    # Create database tables
    create_tables()
    logger.info("Database tables created")
    
    # Initialize core components
    tool_registry = ToolRegistry()
    state_manager = StateManager()
    graph_manager = GraphManager()
    websocket_manager = WebSocketManager()
    execution_engine = ExecutionEngine(
        tool_registry=tool_registry,
        state_manager=state_manager,
        graph_manager=graph_manager,
        max_concurrent_executions=10,
        websocket_manager=websocket_manager
    )
    
    # Register test tools
    from app.tools.test_tools import (
        test_function,
        simple_math_function,
        conditional_function,
        state_logger_function
    )
    
    # Register tools, handling cases where they might already exist
    tools_to_register = [
        ("test_function", test_function, "A simple test function for workflow testing"),
        ("simple_math_function", simple_math_function, "Performs basic math operations on workflow state"),
        ("conditional_function", conditional_function, "Demonstrates conditional logic based on state values"),
        ("state_logger_function", state_logger_function, "Logs current workflow state for debugging")
    ]
    
    for tool_name, tool_func, tool_desc in tools_to_register:
        try:
            if not tool_registry.tool_exists(tool_name):
                tool_registry.register_tool(tool_name, tool_func, tool_desc)
                logger.info(f"Registered tool: {tool_name}")
            else:
                logger.info(f"Tool already exists: {tool_name}")
        except Exception as e:
            logger.warning(f"Failed to register tool {tool_name}: {e}")
    
    logger.info("Test tools registration completed")
    
    # Initialize API dependencies
    init_dependencies(
        graph_manager=graph_manager,
        execution_engine=execution_engine,
        tool_registry=tool_registry,
        state_manager=state_manager,
        websocket_manager=websocket_manager
    )
    
    logger.info("Core components initialized")
    
    # Start WebSocket broadcast processor
    websocket_manager.start_broadcast_processor()
    logger.info("WebSocket broadcast processor started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agent Workflow Engine")
    
    # Stop WebSocket broadcast processor
    try:
        websocket_manager.stop_broadcast_processor()
        logger.info("WebSocket broadcast processor stopped")
    except Exception as e:
        logger.error(f"Error stopping WebSocket broadcast processor: {str(e)}")
    
    # Shutdown execution engine
    try:
        execution_engine.shutdown()
        logger.info("Execution engine shutdown completed")
    except Exception as e:
        logger.error(f"Error during execution engine shutdown: {str(e)}")


# Create FastAPI application
app = FastAPI(
    title="Agent Workflow Engine",
    description="A lightweight workflow engine for defining, executing, and monitoring computational workflows",
    version="1.0.0",
    lifespan=lifespan
)

# Include API router
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "Agent Workflow Engine is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "agent-workflow-engine"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)