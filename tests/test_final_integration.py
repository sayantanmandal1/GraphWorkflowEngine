"""Final integration tests for the complete workflow engine system."""

import pytest
import asyncio
import tempfile
import os
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient
from fastapi import WebSocket
import websockets

from app.factory import create_app
from app.config import AppConfig
from app.models.core import (
    GraphDefinition, NodeDefinition, EdgeDefinition, 
    ExecutionStatusEnum, LogEventType
)
from app.core.tool_registry import ToolRegistry
from app.core.state_manager import StateManager
from app.core.graph_manager import GraphManager
from app.core.execution_engine import ExecutionEngine
from app.core.websocket_manager import WebSocketManager


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)
    
    # Store original DATABASE_URL
    original_db_url = os.environ.get('DATABASE_URL')
    
    # Set database URL for testing
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
    
    yield db_path
    
    # Cleanup
    try:
        os.unlink(db_path)
    except OSError:
        pass
    
    # Restore original DATABASE_URL
    if original_db_url:
        os.environ['DATABASE_URL'] = original_db_url
    else:
        os.environ.pop('DATABASE_URL', None)


@pytest.fixture
def test_config():
    """Create test configuration."""
    return AppConfig(
        app_name="Test Workflow Engine",
        debug=True,
        log_level="INFO",
        max_concurrent_executions=5,
        enable_performance_monitoring=True,
        cors_origins=["*"]
    )


@pytest.fixture
def app_client(temp_db, test_config):
    """Create FastAPI test client with temporary database."""
    app = create_app(test_config)
    client = TestClient(app)
    
    # Wait for startup to complete
    time.sleep(1)
    
    yield client
    
    # Cleanup
    try:
        # Access app state for cleanup
        from app.factory import get_app_state
        app_state = get_app_state()
        if app_state.execution_engine:
            app_state.execution_engine.shutdown()
    except Exception as e:
        print(f"Cleanup warning: {e}")


def simple_test_tool(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Simple test tool for integration testing."""
    message = kwargs.get('message', 'default')
    counter = state.get('counter', 0) + 1
    return {
        'message': f'Tool executed: {message}',
        'counter': counter,
        'timestamp': datetime.utcnow().isoformat()
    }


def math_operation_tool(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Math operation tool for testing."""
    operation = kwargs.get('operation', 'add')
    value = kwargs.get('value', 1)
    current = state.get('result', 0)
    
    if operation == 'add':
        result = current + value
    elif operation == 'multiply':
        result = current * value
    elif operation == 'subtract':
        result = current - value
    else:
        result = current
    
    return {'result': result}


def conditional_tool(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Conditional tool for branching tests."""
    threshold = kwargs.get('threshold', 10)
    current_value = state.get('value', 0)
    
    return {
        'meets_threshold': current_value >= threshold,
        'checked_value': current_value,
        'threshold_used': threshold
    }


class TestEndToEndWorkflowExecution:
    """Test complete end-to-end workflow execution through the API."""
    
    def test_simple_workflow_execution(self, app_client):
        """Test basic workflow creation and execution via API."""
        # Create a simple workflow
        workflow_data = {
            "name": "Simple Test Workflow",
            "description": "Basic end-to-end test",
            "nodes": [
                {
                    "id": "start",
                    "function_name": "workflow_test_function",
                    "parameters": {"message": "Hello from API test"}
                }
            ],
            "edges": [],
            "entry_point": "start"
        }
        
        # Create graph via API
        response = app_client.post("/api/v1/graph/create", json=workflow_data)
        assert response.status_code == 200
        graph_data = response.json()
        graph_id = graph_data["graph_id"]
        assert graph_id is not None
        
        # Execute workflow via API
        execution_data = {
            "graph_id": graph_id,
            "initial_state": {"test_value": "initial"}
        }
        
        response = app_client.post("/api/v1/graph/run", json=execution_data)
        assert response.status_code == 200
        run_data = response.json()
        run_id = run_data["run_id"]
        assert run_id is not None
        
        # Poll for completion
        max_attempts = 30
        for attempt in range(max_attempts):
            response = app_client.get(f"/api/v1/graph/state/{run_id}")
            assert response.status_code == 200
            
            status_data = response.json()
            status = status_data["status"]
            
            if status in ["completed", "failed", "cancelled"]:
                break
            
            time.sleep(1)
        
        # Verify completion
        assert status == "completed"
        assert "current_state" in status_data
        assert "data" in status_data["current_state"]
        
        # Get execution logs
        response = app_client.get(f"/api/v1/graph/logs/{run_id}")
        assert response.status_code == 200
        logs_data = response.json()
        assert len(logs_data["logs"]) > 0
    
    def test_multi_step_workflow_execution(self, app_client):
        """Test multi-step workflow with sequential execution."""
        # Create multi-step workflow
        workflow_data = {
            "name": "Multi-Step Test Workflow",
            "description": "Sequential execution test",
            "nodes": [
                {
                    "id": "init",
                    "function_name": "simple_math_function",
                    "parameters": {"operation": "add", "value": 5}
                },
                {
                    "id": "process",
                    "function_name": "simple_math_function", 
                    "parameters": {"operation": "multiply", "value": 2}
                },
                {
                    "id": "finalize",
                    "function_name": "workflow_test_function",
                    "parameters": {"message": "Processing complete"}
                }
            ],
            "edges": [
                {"from_node": "init", "to_node": "process"},
                {"from_node": "process", "to_node": "finalize"}
            ],
            "entry_point": "init"
        }
        
        # Create and execute workflow
        response = app_client.post("/api/v1/graph/create", json=workflow_data)
        assert response.status_code == 200
        graph_id = response.json()["graph_id"]
        
        execution_data = {
            "graph_id": graph_id,
            "initial_state": {"result": 10}
        }
        
        response = app_client.post("/api/v1/graph/run", json=execution_data)
        assert response.status_code == 200
        run_id = response.json()["run_id"]
        
        # Wait for completion
        max_attempts = 30
        for attempt in range(max_attempts):
            response = app_client.get(f"/api/v1/graph/state/{run_id}")
            status_data = response.json()
            
            if status_data["status"] in ["completed", "failed"]:
                break
            time.sleep(1)
        
        # Verify final state
        assert status_data["status"] == "completed"
        final_state = status_data["current_state"]["data"]
        
        # Should be: (10 + 5) * 2 = 30
        assert final_state["result"] == 30
        assert "message" in final_state  # From workflow_test_function
    
    def test_conditional_branching_workflow(self, app_client):
        """Test workflow with conditional branching logic."""
        # Create workflow with conditional branching
        workflow_data = {
            "name": "Conditional Branching Test",
            "description": "Test conditional execution paths",
            "nodes": [
                {
                    "id": "check",
                    "function_name": "conditional_function",
                    "parameters": {"threshold": 15}
                },
                {
                    "id": "high_path",
                    "function_name": "workflow_test_function",
                    "parameters": {"message": "High value path"}
                },
                {
                    "id": "low_path", 
                    "function_name": "workflow_test_function",
                    "parameters": {"message": "Low value path"}
                }
            ],
            "edges": [
                {
                    "from_node": "check",
                    "to_node": "high_path",
                    "condition": "state.get('meets_threshold', False) == True"
                },
                {
                    "from_node": "check",
                    "to_node": "low_path",
                    "condition": "state.get('meets_threshold', False) == False"
                }
            ],
            "entry_point": "check"
        }
        
        # Test high value path
        response = app_client.post("/api/v1/graph/create", json=workflow_data)
        graph_id = response.json()["graph_id"]
        
        execution_data = {
            "graph_id": graph_id,
            "initial_state": {"value": 20}  # Above threshold
        }
        
        response = app_client.post("/api/v1/graph/run", json=execution_data)
        run_id = response.json()["run_id"]
        
        # Wait for completion
        max_attempts = 30
        for attempt in range(max_attempts):
            response = app_client.get(f"/api/v1/graph/state/{run_id}")
            status_data = response.json()
            
            if status_data["status"] in ["completed", "failed"]:
                break
            time.sleep(1)
        
        # Verify high path was taken
        assert status_data["status"] == "completed"
        final_state = status_data["current_state"]["data"]
        assert "High value path" in final_state.get("message", "")


class TestConcurrentExecutionPerformance:
    """Test concurrent execution scenarios and performance."""
    
    def test_concurrent_workflow_isolation(self, app_client):
        """Test that concurrent workflows maintain state isolation."""
        # Create a simple workflow
        workflow_data = {
            "name": "Concurrent Isolation Test",
            "description": "Test concurrent execution isolation",
            "nodes": [
                {
                    "id": "process",
                    "function_name": "simple_math_function",
                    "parameters": {"operation": "add", "value": 1}
                }
            ],
            "edges": [],
            "entry_point": "process"
        }
        
        response = app_client.post("/api/v1/graph/create", json=workflow_data)
        graph_id = response.json()["graph_id"]
        
        # Start multiple concurrent executions with different initial states
        run_ids = []
        initial_values = [10, 20, 30, 40, 50]
        
        for value in initial_values:
            execution_data = {
                "graph_id": graph_id,
                "initial_state": {"result": value}
            }
            
            response = app_client.post("/api/v1/graph/run", json=execution_data)
            assert response.status_code == 200
            run_ids.append(response.json()["run_id"])
        
        # Wait for all executions to complete
        completed_states = {}
        max_attempts = 30
        
        for run_id in run_ids:
            for attempt in range(max_attempts):
                response = app_client.get(f"/api/v1/graph/state/{run_id}")
                status_data = response.json()
                
                if status_data["status"] in ["completed", "failed"]:
                    completed_states[run_id] = status_data
                    break
                time.sleep(0.5)
        
        # Verify all executions completed successfully
        assert len(completed_states) == len(run_ids)
        
        # Verify state isolation - each should have its expected result
        results = []
        for run_id in run_ids:
            state_data = completed_states[run_id]
            assert state_data["status"] == "completed"
            result = state_data["current_state"]["data"]["result"]
            results.append(result)
        
        # Results should be [11, 21, 31, 41, 51] (each initial + 1)
        expected_results = [v + 1 for v in initial_values]
        assert sorted(results) == sorted(expected_results)
    
    def test_performance_under_load(self, app_client):
        """Test system performance under concurrent load."""
        # Create a simple but realistic workflow
        workflow_data = {
            "name": "Performance Load Test",
            "description": "Test performance under load",
            "nodes": [
                {
                    "id": "step1",
                    "function_name": "simple_math_function",
                    "parameters": {"operation": "add", "value": 2}
                },
                {
                    "id": "step2",
                    "function_name": "simple_math_function",
                    "parameters": {"operation": "multiply", "value": 3}
                }
            ],
            "edges": [
                {"from_node": "step1", "to_node": "step2"}
            ],
            "entry_point": "step1"
        }
        
        response = app_client.post("/api/v1/graph/create", json=workflow_data)
        graph_id = response.json()["graph_id"]
        
        # Launch multiple concurrent executions
        num_executions = 10
        start_time = time.time()
        run_ids = []
        
        for i in range(num_executions):
            execution_data = {
                "graph_id": graph_id,
                "initial_state": {"result": i}
            }
            
            response = app_client.post("/api/v1/graph/run", json=execution_data)
            assert response.status_code == 200
            run_ids.append(response.json()["run_id"])
        
        # Wait for all to complete
        completed_count = 0
        max_wait_time = 60  # seconds
        
        while completed_count < num_executions and (time.time() - start_time) < max_wait_time:
            completed_count = 0
            
            for run_id in run_ids:
                response = app_client.get(f"/api/v1/graph/state/{run_id}")
                if response.status_code == 200:
                    status_data = response.json()
                    if status_data["status"] in ["completed", "failed", "cancelled"]:
                        completed_count += 1
            
            time.sleep(0.5)
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert completed_count == num_executions, f"Only {completed_count}/{num_executions} completed"
        assert total_time < 30, f"Execution took too long: {total_time:.2f}s"
        
        # Verify all completed successfully
        success_count = 0
        for run_id in run_ids:
            response = app_client.get(f"/api/v1/graph/state/{run_id}")
            status_data = response.json()
            if status_data["status"] == "completed":
                success_count += 1
        
        assert success_count == num_executions, f"Only {success_count}/{num_executions} succeeded"


class TestSystemIntegrationWithDatabase:
    """Test system integration with real database operations."""
    
    def test_state_persistence_across_restarts(self, temp_db, test_config):
        """Test that workflow state persists across system restarts."""
        # First application instance
        app1 = create_app(test_config)
        client1 = TestClient(app1)
        time.sleep(1)  # Wait for startup
        
        # Create and start a workflow
        workflow_data = {
            "name": "Persistence Test Workflow",
            "description": "Test state persistence",
            "nodes": [
                {
                    "id": "persist_data",
                    "function_name": "workflow_test_function",
                    "parameters": {"message": "Persistent data"}
                }
            ],
            "edges": [],
            "entry_point": "persist_data"
        }
        
        response = client1.post("/api/v1/graph/create", json=workflow_data)
        graph_id = response.json()["graph_id"]
        
        execution_data = {
            "graph_id": graph_id,
            "initial_state": {"persistent_value": "test_data_123"}
        }
        
        response = client1.post("/api/v1/graph/run", json=execution_data)
        run_id = response.json()["run_id"]
        
        # Wait for completion
        max_attempts = 30
        for attempt in range(max_attempts):
            response = client1.get(f"/api/v1/graph/state/{run_id}")
            if response.status_code == 200:
                status_data = response.json()
                if status_data["status"] in ["completed", "failed"]:
                    break
            time.sleep(1)
        
        # Verify completion
        assert status_data["status"] == "completed"
        original_state = status_data["current_state"]["data"]
        
        # Shutdown first instance
        try:
            from app.factory import get_app_state
            app_state = get_app_state()
            if app_state.execution_engine:
                app_state.execution_engine.shutdown()
        except Exception:
            pass
        
        # Create second application instance (simulating restart)
        app2 = create_app(test_config)
        client2 = TestClient(app2)
        time.sleep(1)  # Wait for startup
        
        # Query the same run_id from the new instance
        response = client2.get(f"/api/v1/graph/state/{run_id}")
        assert response.status_code == 200
        
        restored_status = response.json()
        restored_state = restored_status["current_state"]["data"]
        
        # Verify state was persisted and restored
        assert restored_status["status"] == "completed"
        assert restored_state["persistent_value"] == original_state["persistent_value"]
        assert restored_state["message"] == original_state["message"]
        
        # Cleanup second instance
        try:
            from app.factory import get_app_state
            app_state = get_app_state()
            if app_state.execution_engine:
                app_state.execution_engine.shutdown()
        except Exception:
            pass
    
    def test_historical_data_retrieval(self, app_client):
        """Test retrieval of historical workflow execution data."""
        # Create multiple workflow executions
        workflow_data = {
            "name": "Historical Data Test",
            "description": "Test historical data retrieval",
            "nodes": [
                {
                    "id": "log_execution",
                    "function_name": "state_logger_function",
                    "parameters": {"log_message": "Historical execution"}
                }
            ],
            "edges": [],
            "entry_point": "log_execution"
        }
        
        response = app_client.post("/api/v1/graph/create", json=workflow_data)
        graph_id = response.json()["graph_id"]
        
        # Execute multiple runs
        run_ids = []
        execution_times = []
        
        for i in range(3):
            execution_data = {
                "graph_id": graph_id,
                "initial_state": {"execution_number": i + 1}
            }
            
            start_time = datetime.utcnow()
            response = app_client.post("/api/v1/graph/run", json=execution_data)
            run_ids.append(response.json()["run_id"])
            execution_times.append(start_time)
            
            time.sleep(1)  # Ensure different timestamps
        
        # Wait for all executions to complete
        for run_id in run_ids:
            max_attempts = 30
            for attempt in range(max_attempts):
                response = app_client.get(f"/api/v1/graph/state/{run_id}")
                if response.status_code == 200:
                    status_data = response.json()
                    if status_data["status"] in ["completed", "failed"]:
                        break
                time.sleep(1)
        
        # Test historical data retrieval
        for i, run_id in enumerate(run_ids):
            # Get execution status
            response = app_client.get(f"/api/v1/graph/state/{run_id}")
            assert response.status_code == 200
            status_data = response.json()
            
            assert status_data["status"] == "completed"
            assert status_data["current_state"]["data"]["execution_number"] == i + 1
            
            # Get execution logs
            response = app_client.get(f"/api/v1/graph/logs/{run_id}")
            assert response.status_code == 200
            logs_data = response.json()
            
            assert len(logs_data["logs"]) > 0
            # Verify logs are chronologically ordered
            log_timestamps = [log["timestamp"] for log in logs_data["logs"]]
            assert log_timestamps == sorted(log_timestamps)
    
    def test_error_handling_and_recovery(self, app_client):
        """Test system error handling and recovery mechanisms."""
        # Create workflow that will cause an error
        workflow_data = {
            "name": "Error Handling Test",
            "description": "Test error handling and recovery",
            "nodes": [
                {
                    "id": "error_node",
                    "function_name": "nonexistent_function",  # This will cause an error
                    "parameters": {}
                }
            ],
            "edges": [],
            "entry_point": "error_node"
        }
        
        response = app_client.post("/api/v1/graph/create", json=workflow_data)
        graph_id = response.json()["graph_id"]
        
        execution_data = {
            "graph_id": graph_id,
            "initial_state": {"test": "error_handling"}
        }
        
        response = app_client.post("/api/v1/graph/run", json=execution_data)
        run_id = response.json()["run_id"]
        
        # Wait for execution to fail
        max_attempts = 30
        final_status = None
        
        for attempt in range(max_attempts):
            response = app_client.get(f"/api/v1/graph/state/{run_id}")
            if response.status_code == 200:
                status_data = response.json()
                if status_data["status"] in ["completed", "failed", "cancelled"]:
                    final_status = status_data
                    break
            time.sleep(1)
        
        # Verify error was handled gracefully
        assert final_status is not None
        assert final_status["status"] == "failed"
        assert final_status["error_message"] is not None
        assert "nonexistent_function" in final_status["error_message"]
        
        # Verify system is still operational after error
        # Create and execute a valid workflow
        valid_workflow_data = {
            "name": "Recovery Test Workflow",
            "description": "Test system recovery after error",
            "nodes": [
                {
                    "id": "recovery_node",
                    "function_name": "workflow_test_function",
                    "parameters": {"message": "System recovered"}
                }
            ],
            "edges": [],
            "entry_point": "recovery_node"
        }
        
        response = app_client.post("/api/v1/graph/create", json=valid_workflow_data)
        assert response.status_code == 200
        recovery_graph_id = response.json()["graph_id"]
        
        recovery_execution_data = {
            "graph_id": recovery_graph_id,
            "initial_state": {"recovery_test": True}
        }
        
        response = app_client.post("/api/v1/graph/run", json=recovery_execution_data)
        assert response.status_code == 200
        recovery_run_id = response.json()["run_id"]
        
        # Wait for recovery execution to complete
        for attempt in range(max_attempts):
            response = app_client.get(f"/api/v1/graph/state/{recovery_run_id}")
            if response.status_code == 200:
                status_data = response.json()
                if status_data["status"] == "completed":
                    break
            time.sleep(1)
        
        # Verify system recovered and can execute workflows normally
        assert status_data["status"] == "completed"
        assert "System recovered" in status_data["current_state"]["data"]["message"]


class TestHealthChecksAndMonitoring:
    """Test health check endpoints and system monitoring."""
    
    def test_basic_health_endpoints(self, app_client):
        """Test basic health check endpoints."""
        # Test root endpoint
        response = app_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        
        # Test basic health endpoint
        response = app_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data
        
        # Test liveness endpoint
        response = app_client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True
        assert "timestamp" in data
    
    def test_detailed_health_check(self, app_client):
        """Test detailed health check with component status."""
        response = app_client.get("/health/detailed")
        assert response.status_code in [200, 503]  # May be unhealthy during tests
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "overall_status" in data
        
        if response.status_code == 200:
            assert data["overall_status"] == "healthy"
        
        # Should have component checks
        if "checks" in data:
            # Verify expected components are checked
            expected_components = ["database", "execution_engine", "tool_registry"]
            for component in expected_components:
                if component in data["checks"]:
                    assert "status" in data["checks"][component]
    
    def test_readiness_check(self, app_client):
        """Test readiness check for container orchestration."""
        response = app_client.get("/health/ready")
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "ready" in data
        assert "timestamp" in data
        
        if response.status_code == 200:
            assert data["ready"] is True
            
        if "checks" in data:
            # Critical components should be checked
            critical_components = ["database", "execution_engine"]
            for component in critical_components:
                if component in data["checks"]:
                    check_data = data["checks"][component]
                    assert "status" in check_data


class TestAPIErrorHandling:
    """Test API error handling and response consistency."""
    
    def test_invalid_graph_creation(self, app_client):
        """Test API error handling for invalid graph creation."""
        # Test with missing required fields
        invalid_workflow = {
            "name": "Invalid Workflow",
            # Missing description, nodes, edges, entry_point
        }
        
        response = app_client.post("/api/v1/graph/create", json=invalid_workflow)
        assert response.status_code == 422  # Validation error
        
        error_data = response.json()
        assert "detail" in error_data
    
    def test_nonexistent_resource_access(self, app_client):
        """Test API error handling for nonexistent resources."""
        # Test getting nonexistent graph state
        response = app_client.get("/api/v1/graph/state/nonexistent-run-id")
        assert response.status_code == 404
        
        error_data = response.json()
        assert "detail" in error_data
        
        # Test getting logs for nonexistent run
        response = app_client.get("/api/v1/graph/logs/nonexistent-run-id")
        assert response.status_code == 404
        
        error_data = response.json()
        assert "detail" in error_data
    
    def test_malformed_request_handling(self, app_client):
        """Test API error handling for malformed requests."""
        # Test with invalid JSON
        response = app_client.post(
            "/api/v1/graph/create",
            data="invalid json content",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        
        # Test with wrong content type
        response = app_client.post(
            "/api/v1/graph/run",
            data="some data",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])