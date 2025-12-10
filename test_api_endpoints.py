"""Test script for the FastAPI REST endpoints."""

import requests
import json
import time
from typing import Dict, Any


def test_api_endpoints():
    """Test the FastAPI REST endpoints."""
    base_url = "http://localhost:8000"
    api_base = f"{base_url}/api/v1"
    
    print("Testing Agent Workflow Engine API Endpoints")
    print("=" * 50)
    
    # Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test graph creation
    print("\n2. Testing graph creation...")
    
    # Create a simple test graph
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
                    "id": "end",
                    "function_name": "test_function",
                    "parameters": {"message": "Goodbye"}
                }
            ],
            "edges": [
                {
                    "from_node": "start",
                    "to_node": "end"
                }
            ],
            "entry_point": "start",
            "exit_conditions": []
        }
    }
    
    try:
        response = requests.post(
            f"{api_base}/graph/create",
            json=test_graph,
            headers={"Content-Type": "application/json"}
        )
        print(f"Graph creation: {response.status_code}")
        if response.status_code == 201:
            graph_data = response.json()
            print(f"Created graph ID: {graph_data['graph_id']}")
            graph_id = graph_data['graph_id']
        else:
            print(f"Error: {response.text}")
            return
    except Exception as e:
        print(f"Graph creation failed: {e}")
        return
    
    # Test graph listing
    print("\n3. Testing graph listing...")
    try:
        response = requests.get(f"{api_base}/graphs")
        print(f"Graph listing: {response.status_code}")
        if response.status_code == 200:
            graphs = response.json()
            print(f"Found {len(graphs)} graphs")
            for graph in graphs:
                print(f"  - {graph['name']} ({graph['id']})")
    except Exception as e:
        print(f"Graph listing failed: {e}")
    
    # Test graph retrieval
    print("\n4. Testing graph retrieval...")
    try:
        response = requests.get(f"{api_base}/graph/{graph_id}")
        print(f"Graph retrieval: {response.status_code}")
        if response.status_code == 200:
            graph = response.json()
            print(f"Retrieved graph: {graph['name']}")
    except Exception as e:
        print(f"Graph retrieval failed: {e}")
    
    # Test graph validation
    print("\n5. Testing graph validation...")
    try:
        response = requests.post(
            f"{api_base}/graph/validate",
            json=test_graph,
            headers={"Content-Type": "application/json"}
        )
        print(f"Graph validation: {response.status_code}")
        if response.status_code == 200:
            validation = response.json()
            print(f"Validation result: Valid={validation['is_valid']}")
            if validation['errors']:
                print(f"Errors: {validation['errors']}")
            if validation['warnings']:
                print(f"Warnings: {validation['warnings']}")
    except Exception as e:
        print(f"Graph validation failed: {e}")
    
    # Test workflow execution (this will fail because we don't have the test_function registered)
    print("\n6. Testing workflow execution...")
    try:
        execution_request = {
            "graph_id": graph_id,
            "initial_state": {"test_data": "initial value"}
        }
        response = requests.post(
            f"{api_base}/graph/run",
            json=execution_request,
            headers={"Content-Type": "application/json"}
        )
        print(f"Workflow execution: {response.status_code}")
        if response.status_code == 202:
            run_data = response.json()
            print(f"Started execution with run ID: {run_data['run_id']}")
            run_id = run_data['run_id']
            
            # Test execution status
            print("\n7. Testing execution status...")
            time.sleep(1)  # Give it a moment
            status_response = requests.get(f"{api_base}/graph/state/{run_id}")
            print(f"Execution status: {status_response.status_code}")
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"Status: {status['status']}")
                if status.get('error_message'):
                    print(f"Error: {status['error_message']}")
            
            # Test execution logs
            print("\n8. Testing execution logs...")
            logs_response = requests.get(f"{api_base}/graph/logs/{run_id}")
            print(f"Execution logs: {logs_response.status_code}")
            if logs_response.status_code == 200:
                logs = logs_response.json()
                print(f"Found {len(logs)} log entries")
                for log in logs[:3]:  # Show first 3 logs
                    print(f"  - {log['timestamp']}: {log['event_type']} - {log['message']}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Workflow execution failed: {e}")
    
    # Test error cases
    print("\n9. Testing error cases...")
    
    # Test non-existent graph
    try:
        response = requests.get(f"{api_base}/graph/non-existent-id")
        print(f"Non-existent graph: {response.status_code}")
        if response.status_code == 404:
            print("✓ Correctly returned 404 for non-existent graph")
    except Exception as e:
        print(f"Error case test failed: {e}")
    
    # Test invalid graph creation
    try:
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
            headers={"Content-Type": "application/json"}
        )
        print(f"Invalid graph creation: {response.status_code}")
        if response.status_code == 400:
            print("✓ Correctly returned 400 for invalid graph")
    except Exception as e:
        print(f"Invalid graph test failed: {e}")
    
    print("\n" + "=" * 50)
    print("API endpoint testing completed!")


if __name__ == "__main__":
    print("Make sure the server is running with: python -m app.main")
    print("Press Enter to start testing...")
    input()
    test_api_endpoints()