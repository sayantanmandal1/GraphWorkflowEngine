"""Test tools for workflow engine testing."""

from typing import Dict, Any, Optional
from ..core.logging import get_logger

logger = get_logger(__name__)


def workflow_test_function(state: Dict[str, Any], context: Optional[Any] = None, message: str = "Default message", **kwargs) -> Dict[str, Any]:
    """
    A simple test function for workflow testing.
    
    Args:
        state: Current workflow state
        context: Execution context (optional)
        message: Message to log and store
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with updated state data
    """
    logger.info(f"Test function called with message: {message}")
    
    # Update state with test data
    result = {
        "test_message": message,
        "execution_count": state.get("execution_count", 0) + 1,
        "last_executed": "workflow_test_function"
    }
    
    # Log the execution
    logger.debug(f"Test function updating state: {result}")
    
    return result


def simple_math_function(state: Dict[str, Any], context: Optional[Any] = None, operation: str = "add", value: float = 1.0, **kwargs) -> Dict[str, Any]:
    """
    A simple math function for testing calculations.
    
    Args:
        state: Current workflow state
        context: Execution context (optional)
        operation: Math operation to perform (add, subtract, multiply, divide)
        value: Value to use in the operation
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with calculation result
    """
    logger.info(f"Math function called with operation: {operation}, value: {value}")
    
    current_value = state.get("result", 0.0)
    
    if operation == "add":
        new_value = current_value + value
    elif operation == "subtract":
        new_value = current_value - value
    elif operation == "multiply":
        new_value = current_value * value
    elif operation == "divide":
        if value == 0:
            logger.error("Division by zero attempted")
            raise ValueError("Cannot divide by zero")
        new_value = current_value / value
    else:
        logger.warning(f"Unknown operation: {operation}, defaulting to add")
        new_value = current_value + value
    
    result = {
        "result": new_value,
        "last_operation": operation,
        "last_value": value
    }
    
    logger.debug(f"Math function result: {result}")
    
    return result


def conditional_function(state: Dict[str, Any], context: Optional[Any] = None, threshold: float = 10.0, **kwargs) -> Dict[str, Any]:
    """
    A function that demonstrates conditional logic.
    
    Args:
        state: Current workflow state
        context: Execution context (optional)
        threshold: Threshold value for comparison
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with conditional result
    """
    current_value = state.get("result", 0.0)
    
    if current_value > threshold:
        status = "above_threshold"
        message = f"Value {current_value} is above threshold {threshold}"
    elif current_value < -threshold:
        status = "below_negative_threshold"
        message = f"Value {current_value} is below negative threshold {-threshold}"
    else:
        status = "within_threshold"
        message = f"Value {current_value} is within threshold range ±{threshold}"
    
    logger.info(message)
    
    result = {
        "threshold_status": status,
        "threshold_message": message,
        "threshold_value": threshold,
        "checked_value": current_value
    }
    
    return result


def state_logger_function(state: Dict[str, Any], context: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
    """
    A function that logs the current state for debugging.
    
    Args:
        state: Current workflow state
        context: Execution context (optional)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with logging information
    """
    logger.info(f"Current workflow state: {state}")
    
    state_summary = {
        "state_keys": list(state.keys()),
        "state_size": len(state),
        "logged_at": "state_logger_function"
    }
    
    # Add some metadata about the state
    for key, value in state.items():
        if isinstance(value, (int, float)):
            state_summary[f"{key}_type"] = "numeric"
        elif isinstance(value, str):
            state_summary[f"{key}_type"] = "string"
        elif isinstance(value, (list, tuple)):
            state_summary[f"{key}_type"] = "sequence"
        elif isinstance(value, dict):
            state_summary[f"{key}_type"] = "mapping"
        else:
            state_summary[f"{key}_type"] = "other"
    
    return state_summary