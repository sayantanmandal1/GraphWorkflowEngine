"""Middleware for error handling and logging."""

import time
import uuid
from datetime import datetime
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .exceptions import WorkflowEngineError, create_error_response
from .logging import get_logger, set_logging_context, clear_logging_context


logger = get_logger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive error handling and logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with error handling."""
        # Generate request ID for tracing
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Set logging context
        set_logging_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown"
        )
        
        try:
            # Log request start
            logger.info(f"Request started: {request.method} {request.url.path}")
            
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log successful response
            logger.info(
                f"Request completed: {request.method} {request.url.path} - "
                f"Status: {response.status_code} - Duration: {duration:.3f}s"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except WorkflowEngineError as e:
            # Handle known workflow engine errors
            duration = time.time() - start_time
            
            logger.warning(
                f"Workflow engine error: {request.method} {request.url.path} - "
                f"Error: {e.error_code} - Duration: {duration:.3f}s",
                extra={"error_details": e.to_dict()}
            )
            
            # Determine HTTP status code
            status_code = self._get_status_code_for_error(e)
            
            return JSONResponse(
                status_code=status_code,
                content=create_error_response(e),
                headers={"X-Request-ID": request_id}
            )
            
        except Exception as e:
            # Handle unexpected errors
            duration = time.time() - start_time
            
            logger.error(
                f"Unexpected error: {request.method} {request.url.path} - "
                f"Error: {str(e)} - Duration: {duration:.3f}s",
                exc_info=True
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "InternalServerError",
                    "message": "An unexpected error occurred",
                    "details": {
                        "error_type": type(e).__name__,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "request_id": request_id
                },
                headers={"X-Request-ID": request_id}
            )
            
        finally:
            # Clear logging context
            clear_logging_context()
    
    def _get_status_code_for_error(self, error: WorkflowEngineError) -> int:
        """Determine appropriate HTTP status code for workflow engine error."""
        from .exceptions import (
            GraphValidationError, StorageError, ExecutionEngineError,
            ResourceExhaustionError, TransientError, ConfigurationError
        )
        
        if isinstance(error, GraphValidationError):
            return 400
        elif isinstance(error, StorageError):
            if "not found" in error.message.lower():
                return 404
            return 500
        elif isinstance(error, ResourceExhaustionError):
            return 503
        elif isinstance(error, TransientError):
            return 503
        elif isinstance(error, ConfigurationError):
            return 500
        elif isinstance(error, ExecutionEngineError):
            if "queue is full" in error.message.lower():
                return 503
            return 500
        else:
            return 500


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details."""
        start_time = time.time()
        
        # Log request details
        logger.debug(
            f"Request details: {request.method} {request.url} - "
            f"Headers: {dict(request.headers)} - "
            f"Query params: {dict(request.query_params)}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response details
        logger.debug(
            f"Response details: Status {response.status_code} - "
            f"Duration: {duration:.3f}s - "
            f"Headers: {dict(response.headers)}"
        )
        
        return response


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and alerting."""
    
    def __init__(self, app, slow_request_threshold: float = 5.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor request performance."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Check for slow requests
        if duration > self.slow_request_threshold:
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path} - "
                f"Duration: {duration:.3f}s (threshold: {self.slow_request_threshold}s)"
            )
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response