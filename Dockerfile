# Multi-stage build for Agent Workflow Engine
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r workflow && useradd -r -g workflow workflow

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/workflow/.local

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY .env.example .env

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R workflow:workflow /app

# Switch to non-root user
USER workflow

# Set environment variables
ENV PATH=/home/workflow/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV WORKFLOW_ENGINE_LOG_FILE=/app/logs/workflow_engine.log
ENV WORKFLOW_ENGINE_DATABASE_URL=sqlite:///./data/workflow_engine.db

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health/live')" || exit 1

# Default command
CMD ["python", "-m", "app.startup", "run"]