"""Main FastAPI application for the workflow engine."""

from app.factory import create_app
from app.config import get_config

# Create the FastAPI application using the factory pattern
app = create_app()

if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn_config = config.get_uvicorn_config()
    uvicorn.run(app, **uvicorn_config)