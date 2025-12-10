"""Main FastAPI application for the workflow engine."""

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.logging import setup_logging, get_logger
from app.storage.database import create_tables


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
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agent Workflow Engine")


# Create FastAPI application
app = FastAPI(
    title="Agent Workflow Engine",
    description="A lightweight workflow engine for defining, executing, and monitoring computational workflows",
    version="1.0.0",
    lifespan=lifespan
)


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