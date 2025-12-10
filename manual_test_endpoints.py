"""Manual test for the FastAPI REST endpoints using requests."""

import requests
import json
import time
import sys


def test_endpoints():
    """Test the FastAPI REST endpoints manually."""
    base_url = "http://localhost:8000"
    api_base = f"{base_url}/api/v1"
    
    print("Testing Agent Workflow Engine API Endpoints")
    print("=" * 50)
    print("Make sure the server is running with: python -m app.main")
    print()
    
    try:
        # Test health check
        print("1. Testing health check...")
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✓ Health check passed")
        
        # Test graph creation
        print("\n2. Testing graph creation...")
        import uuid
        unique_name = f"test-workflow-{str(uuid.uuid4())[:8]}"
        test_graph = {
            "graph": {
                "name": unique_name,
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
        
        response = requests.post(
            f"{api_base}/graph/create",
            json=test_graph,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 201
        
        graph_data = response.json()
        graph_id = graph_data['graph_id']
        print(f"   ✓ Graph created with ID: {graph_id}")
        
        # Test graph listing
        print("\n3. Testing graph listing...")
        response = requests.get(f"{api_base}/graphs", timeout=5)
        print(f"   Status: {response.status_code}")
        graphs = response.json()
        print(f"   Found {len(graphs)} graphs")
        assert response.status_code == 200
        assert len(graphs) >= 1
        print("   ✓ Graph listing passed")
        
        # Test graph retrieval
        print("\n4. Testing graph retrieval...")
        response = requests.get(f"{api_base}/graph/{graph_id}", timeout=5)
        print(f"   Status: {response.status_code}")
        assert response.status_code == 200
        graph = response.json()
        assert graph['name'] == unique_name
        print("   ✓ Graph retrieval passed")
        
        # Test graph validation
        print("\n5. Testing graph validation...")
        response = requests.post(
            f"{api_base}/graph/validate",
            json=test_graph,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        validation = response.json()
        print(f"   Valid: {validation['is_valid']}")
        assert response.status_code == 200
        assert validation['is_valid'] is True
        print("   ✓ Graph validation passed")
        
        # Test workflow execution
        print("\n6. Testing workflow execution...")
        execution_request = {
            "graph_id": graph_id,
            "initial_state": {"result": 10.0, "test_data": "initial value"}
        }
        
        response = requests.post(
            f"{api_base}/graph/run",
            json=execution_request,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 202
        
        run_data = response.json()
        run_id = run_data['run_id']
        print(f"   ✓ Workflow started with run ID: {run_id}")
        
        # Test execution status
        print("\n7. Testing execution status...")
        time.sleep(3)  # Give execution time to complete
        response = requests.get(f"{api_base}/graph/state/{run_id}", timeout=5)
        print(f"   Status: {response.status_code}")
        assert response.status_code == 200
        status = response.json()
        print(f"   Execution status: {status['status']}")
        if status.get('error_message'):
            print(f"   Error: {status['error_message']}")
        print("   ✓ Execution status retrieval passed")
        
        # Test execution logs
        print("\n8. Testing execution logs...")
        response = requests.get(f"{api_base}/graph/logs/{run_id}", timeout=5)
        print(f"   Status: {response.status_code}")
        assert response.status_code == 200
        logs = response.json()
        print(f"   Found {len(logs)} log entries")
        for i, log in enumerate(logs[:3]):  # Show first 3 logs
            print(f"   Log {i+1}: {log['event_type']} - {log['message']}")
        print("   ✓ Execution logs retrieval passed")
        
        # Test error cases
        print("\n9. Testing error cases...")
        
        # Non-existent graph
        response = requests.get(f"{api_base}/graph/non-existent-id", timeout=5)
        print(f"   Non-existent graph status: {response.status_code}")
        assert response.status_code == 404
        print("   ✓ Non-existent graph error handling passed")
        
        # Invalid graph creation
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
        
        response = requests.post(
            f"{api_base}/graph/create",
            json=invalid_graph,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        print(f"   Invalid graph creation status: {response.status_code}")
        assert response.status_code in [400, 422]  # 422 for Pydantic validation errors
        print("   ✓ Invalid graph error handling passed")
        
        print("\n" + "=" * 50)
        print("🎉 All API endpoint tests passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the server is running")
        print("   Start the server with: python -m app.main")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timeout: Server may be overloaded")
        return False
    except AssertionError as e:
        print(f"❌ Test assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_endpoints()
    sys.exit(0 if success else 1)