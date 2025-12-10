"""Comprehensive test for the FastAPI REST endpoints."""

import pytest
from fastapi.testclient import TestClient

# Import app and create test client
from app.main import app
client = TestClient(app)


def test_health_endpoints():
    """Test health check endpoints."""
    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Agent Workflow Engine is running"}
    
    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "agent-workflow-engine"


def test_graph_creation_and_retrieval():
    """Test graph creation and retrieval endpoints."""
    # Create a test graph
    test_graph = {
        "graph": {
            "name": "test-workflow",
            "description": "A simple test workflow",
            "nodes": [
                {
                    "id": "start",
                    "function_name": "test_function",
                    "parameters": {"message": "Hello World"}
                },
                {
                    "id": "math",
                    "function_name": "simple_math_function",
                    "parameters": {"operation": "add", "value": 5.0}
                },
                {
                    "id": "end",
                    "function_name": "state_logger_function",
                    "parameters": {}
                }
            ],
            "edges": [
                {"from_node": "start", "to_node": "math"},
                {"from_node": "math", "to_node": "end"}
            ],
            "entry_point": "start",
            "exit_conditions": []
        }
    }
    
    # Test graph creation
    response = client.post("/api/v1/graph/create", json=test_graph)
    assert response.status_code == 201
    
    creation_data = response.json()
    assert "graph_id" in creation_data
    assert creation_data["message"] == "Graph 'test-workflow' created successfully"
    
    graph_id = creation_data["graph_id"]
    
    # Test graph retrieval
    response = client.get(f"/api/v1/graph/{graph_id}")
    assert response.status_code == 200
    
    graph_data = response.json()
    assert graph_data["name"] == "test-workflow"
    assert graph_data["description"] == "A simple test workflow"
    assert len(graph_data["nodes"]) == 3
    assert len(graph_data["edges"]) == 2
    
    return graph_id


def test_graph_listing():
    """Test graph listing endpoint."""
    response = client.get("/api/v1/graphs")
    assert response.status_code == 200
    
    graphs = response.json()
    assert isinstance(graphs, list)
    assert len(graphs) >= 1  # At least the test graph we created
    
    # Check graph structure
    for graph in graphs:
        assert "id" in graph
        assert "name" in graph
        assert "description" in graph
        assert "created_at" in graph
        assert "node_count" in graph


def test_graph_validation():
    """Test graph validation endpoint."""
    # Test valid graph
    valid_graph = {
        "graph": {
            "name": "valid-test",
            "description": "A valid test graph",
            "nodes": [
                {
                    "id": "single_node",
                    "function_name": "test_function",
                    "parameters": {"message": "Valid"}
                }
            ],
            "edges": [],
            "entry_point": "single_node",
            "exit_conditions": []
        }
    }
    
    response = client.post("/api/v1/graph/validate", json=valid_graph)
    assert response.status_code == 200
    
    validation_data = response.json()
    assert validation_data["is_valid"] is True
    assert isinstance(validation_data["errors"], list)
    assert isinstance(validation_data["warnings"], list)
    
    # Test invalid graph
    invalid_graph = {
        "graph": {
            "name": "",  # Invalid empty name
            "description": "Invalid graph",
            "nodes": [],  # No nodes
            "edges": [],
            "entry_point": "non-existent",
            "exit_conditions": []
        }
    }
    
    response = client.post("/api/v1/graph/validate", json=invalid_graph)
    assert response.status_code == 200
    
    validation_data = response.json()
    assert validation_data["is_valid"] is False
    assert len(validation_data["errors"]) > 0


def test_workflow_execution():
    """Test workflow execution endpoints."""
    # First create a graph
    graph_id = test_graph_creation_and_retrieval()
    
    # Test workflow execution
    execution_request = {
        "graph_id": graph_id,
        "initial_state": {"result": 10.0, "test_data": "initial value"}
    }
    
    response = client.post("/api/v1/graph/run", json=execution_request)
    assert response.status_code == 202
    
    execution_data = response.json()
    assert "run_id" in execution_data
    assert execution_data["message"] == "Workflow execution started successfully"
    assert execution_data["status"] == "pending"
    
    run_id = execution_data["run_id"]
    
    # Test execution status
    import time
    time.sleep(2)  # Give execution time to complete
    
    response = client.get(f"/api/v1/graph/state/{run_id}")
    assert response.status_code == 200
    
    status_data = response.json()
    assert status_data["run_id"] == run_id
    assert status_data["graph_id"] == graph_id
    assert "status" in status_data
    assert "current_state" in status_data
    assert "started_at" in status_data
    
    # Test execution logs
    response = client.get(f"/api/v1/graph/logs/{run_id}")
    assert response.status_code == 200
    
    logs = response.json()
    assert isinstance(logs, list)
    assert len(logs) > 0
    
    # Check log structure
    for log in logs:
        assert "timestamp" in log
        assert "run_id" in log
        assert "event_type" in log
        assert "message" in log
    
    return run_id


def test_error_cases():
    """Test error handling in endpoints."""
    # Test non-existent graph retrieval
    response = client.get("/api/v1/graph/non-existent-id")
    assert response.status_code == 404
    
    error_data = response.json()
    assert "error" in error_data["detail"]
    assert error_data["detail"]["error"] == "GraphNotFound"
    
    # Test non-existent run status
    response = client.get("/api/v1/graph/state/non-existent-run-id")
    assert response.status_code == 404
    
    error_data = response.json()
    assert "error" in error_data["detail"]
    assert error_data["detail"]["error"] == "RunNotFound"
    
    # Test non-existent run logs
    response = client.get("/api/v1/graph/logs/non-existent-run-id")
    assert response.status_code == 404
    
    # Test invalid graph creation
    invalid_graph = {
        "graph": {
            "name": "",  # Invalid empty name
            "description": "Invalid graph",
            "nodes": [],  # No nodes
            "edges": [],
            "entry_point": "non-existent",
            "exit_conditions": []
        }
    }
    
    response = client.post("/api/v1/graph/create", json=invalid_graph)
    assert response.status_code == 400
    
    error_data = response.json()
    assert "error" in error_data["detail"]
    assert error_data["detail"]["error"] == "ValidationError"
    
    # Test execution with non-existent graph
    execution_request = {
        "graph_id": "non-existent-graph-id",
        "initial_state": {}
    }
    
    response = client.post("/api/v1/graph/run", json=execution_request)
    assert response.status_code == 404
    
    error_data = response.json()
    assert "error" in error_data["detail"]
    assert error_data["detail"]["error"] == "GraphNotFound"


def test_graph_deletion():
    """Test graph deletion endpoint."""
    # Create a graph to delete
    test_graph = {
        "graph": {
            "name": "delete-test",
            "description": "Graph for deletion test",
            "nodes": [
                {
                    "id": "test_node",
                    "function_name": "test_function",
                    "parameters": {"message": "Delete me"}
                }
            ],
            "edges": [],
            "entry_point": "test_node",
            "exit_conditions": []
        }
    }
    
    response = client.post("/api/v1/graph/create", json=test_graph)
    assert response.status_code == 201
    graph_id = response.json()["graph_id"]
    
    # Delete the graph
    response = client.delete(f"/api/v1/graph/{graph_id}")
    assert response.status_code == 204
    
    # Verify graph is deleted
    response = client.get(f"/api/v1/graph/{graph_id}")
    assert response.status_code == 404
    
    # Test deleting non-existent graph
    response = client.delete("/api/v1/graph/non-existent-id")
    assert response.status_code == 404


def test_execution_cancellation():
    """Test execution cancellation endpoint."""
    # Create and start a workflow
    graph_id = test_graph_creation_and_retrieval()
    
    execution_request = {
        "graph_id": graph_id,
        "initial_state": {"result": 1.0}
    }
    
    response = client.post("/api/v1/graph/run", json=execution_request)
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    
    # Try to cancel (may or may not succeed depending on timing)
    response = client.post(f"/api/v1/graph/cancel/{run_id}")
    assert response.status_code == 200
    
    cancel_data = response.json()
    assert "message" in cancel_data
    assert "cancelled" in cancel_data


if __name__ == "__main__":
    print("Running comprehensive endpoint tests...")
    
    try:
        test_health_endpoints()
        print("✓ Health endpoints test passed")
        
        test_graph_creation_and_retrieval()
        print("✓ Graph creation and retrieval test passed")
        
        test_graph_listing()
        print("✓ Graph listing test passed")
        
        test_graph_validation()
        print("✓ Graph validation test passed")
        
        test_workflow_execution()
        print("✓ Workflow execution test passed")
        
        test_error_cases()
        print("✓ Error cases test passed")
        
        test_graph_deletion()
        print("✓ Graph deletion test passed")
        
        test_execution_cancellation()
        print("✓ Execution cancellation test passed")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()