"""Error recovery mechanisms for transient failures."""

import asyncio
import time
import random
from typing import Callable, Any, Optional, Dict, List, Type
from functools import wraps
from datetime import datetime, timedelta

from .exceptions import (
    WorkflowEngineError, TransientError, StorageError, 
    ResourceExhaustionError, ErrorSeverity
)
from .logging import get_logger, ErrorRecoveryLogger


logger = get_logger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [
            TransientError, StorageError, ResourceExhaustionError
        ]
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should be retried."""
        if attempt >= self.max_attempts:
            return False
        
        # Check if exception is retryable
        if isinstance(exception, WorkflowEngineError):
            return exception.recoverable
        
        return any(isinstance(exception, exc_type) for exc_type in self.retryable_exceptions)
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt."""
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add jitter to prevent thundering herd
            delay *= (0.5 + random.random() * 0.5)
        
        return delay


class CircuitBreaker:
    """Circuit breaker pattern implementation for preventing cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = "closed"  # closed, open, half-open
        
        self.recovery_logger = ErrorRecoveryLogger("circuit_breaker")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
    
    def _call_with_circuit_breaker(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self._state == "open":
            if self._should_attempt_reset():
                self._state = "half-open"
                logger.info("Circuit breaker transitioning to half-open state")
            else:
                raise TransientError(
                    f"Circuit breaker is open. Service unavailable until {self._last_failure_time + timedelta(seconds=self.recovery_timeout)}"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if self._last_failure_time is None:
            return True
        return datetime.utcnow() >= self._last_failure_time + timedelta(seconds=self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful execution."""
        if self._state == "half-open":
            self._state = "closed"
            self._failure_count = 0
            logger.info("Circuit breaker reset to closed state")
    
    def _on_failure(self):
        """Handle failed execution."""
        self._failure_count += 1
        self._last_failure_time = datetime.utcnow()
        
        if self._failure_count >= self.failure_threshold:
            self._state = "open"
            self.recovery_logger.log_recovery_failure(
                "circuit_breaker_opened",
                TransientError(f"Circuit breaker opened after {self._failure_count} failures"),
                self._failure_count
            )


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator to add retry logic to functions."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return _execute_with_retry(func, config, *args, **kwargs)
        return wrapper
    
    return decorator


def _execute_with_retry(func: Callable, config: RetryConfig, *args, **kwargs) -> Any:
    """Execute function with retry logic."""
    recovery_logger = ErrorRecoveryLogger(func.__name__)
    last_exception = None
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if not config.should_retry(e, attempt):
                recovery_logger.log_recovery_failure(
                    func.__name__, e, attempt
                )
                raise
            
            if attempt < config.max_attempts:
                delay = config.get_delay(attempt)
                recovery_logger.log_recovery_attempt(
                    func.__name__, e, attempt, config.max_attempts
                )
                time.sleep(delay)
    
    # If we get here, all attempts failed
    recovery_logger.log_recovery_failure(
        func.__name__, last_exception, config.max_attempts
    )
    raise last_exception


async def with_async_retry(config: Optional[RetryConfig] = None):
    """Decorator to add async retry logic to functions."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await _execute_async_with_retry(func, config, *args, **kwargs)
        return wrapper
    
    return decorator


async def _execute_async_with_retry(func: Callable, config: RetryConfig, *args, **kwargs) -> Any:
    """Execute async function with retry logic."""
    recovery_logger = ErrorRecoveryLogger(func.__name__)
    last_exception = None
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if not config.should_retry(e, attempt):
                recovery_logger.log_recovery_failure(
                    func.__name__, e, attempt
                )
                raise
            
            if attempt < config.max_attempts:
                delay = config.get_delay(attempt)
                recovery_logger.log_recovery_attempt(
                    func.__name__, e, attempt, config.max_attempts
                )
                await asyncio.sleep(delay)
    
    # If we get here, all attempts failed
    recovery_logger.log_recovery_failure(
        func.__name__, last_exception, config.max_attempts
    )
    raise last_exception


class HealthChecker:
    """Health checker for system components."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, Dict[str, Any]] = {}
        self.logger = get_logger("health_checker")
    
    def register_check(self, name: str, check_func: Callable, timeout: float = 5.0):
        """Register a health check function."""
        self.checks[name] = {
            "func": check_func,
            "timeout": timeout
        }
        self.logger.info(f"Registered health check: {name}")
    
    async def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.checks:
            return {
                "status": "error",
                "message": f"Health check '{name}' not found",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        check_info = self.checks[name]
        start_time = time.time()
        
        try:
            # Run check with timeout
            if asyncio.iscoroutinefunction(check_info["func"]):
                result = await asyncio.wait_for(
                    check_info["func"](), 
                    timeout=check_info["timeout"]
                )
            else:
                result = check_info["func"]()
            
            duration = time.time() - start_time
            
            check_result = {
                "status": "healthy",
                "message": result if isinstance(result, str) else "Check passed",
                "duration_ms": round(duration * 1000, 2),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if isinstance(result, dict):
                check_result.update(result)
            
            self.last_results[name] = check_result
            return check_result
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            result = {
                "status": "timeout",
                "message": f"Health check timed out after {check_info['timeout']}s",
                "duration_ms": round(duration * 1000, 2),
                "timestamp": datetime.utcnow().isoformat()
            }
            self.last_results[name] = result
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            result = {
                "status": "unhealthy",
                "message": str(e),
                "error_type": type(e).__name__,
                "duration_ms": round(duration * 1000, 2),
                "timestamp": datetime.utcnow().isoformat()
            }
            self.last_results[name] = result
            return result
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_status = "healthy"
        
        for name in self.checks:
            result = await self.run_check(name)
            results[name] = result
            
            if result["status"] != "healthy":
                overall_status = "unhealthy"
        
        return {
            "overall_status": overall_status,
            "checks": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_last_results(self) -> Dict[str, Any]:
        """Get the last health check results."""
        return {
            "checks": self.last_results,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global health checker instance
health_checker = HealthChecker()