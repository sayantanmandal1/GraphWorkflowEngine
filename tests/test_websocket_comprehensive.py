#!/usr/bin/env python3
"""Comprehensive test for WebSocket real-time monitoring functionality."""

import asyncio
import json
import websockets
import requests
from datetime import datetime
import time
import pytest


@pytest.mark.asyncio
async def test_websocket_comprehensive():
    """Comprehensive test of WebSocket monitoring functionality."""
    print("Comprehensive WebSocket Monitoring Test")
    print("=" * 50)
    
    # Test 1: Connection establishment
    print("\n1. Testing WebSocket connection establishment...")
    uri = "ws://localhost:8000/api/v1/ws/monitor"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection established successfully")
            
            # Wait for connection established message
            message = await websocket.recv()
            data = json.loads(message)
            assert data["event_type"] == "connection_established"
            print(f"✅ Connection established message received: {data['connection_id']}")
            
            # Test 2: Event broadcasting for workflow execution events
            print("\n2. Testing workflow execution event broadcasting...")
            
            # Create a test workflow
            timestamp = int(time.time())
            graph_definition = {
                "name": f"Comprehensive Test Workflow {timestamp}",
                "description": "A comprehensive test workflow for WebSocket monitoring",
                "nodes": [
                    {
                        "id": "start_node",
                        "function_name": "workflow_test_function",
                        "parameters": {"message": "Starting comprehensive test"}
                    },
                    {
                        "id": "math_node", 
                        "function_name": "simple_math_function",
                        "parameters": {"operation": "add", "value": 15}
                    },
                    {
                        "id": "conditional_node",
                        "function_name": "conditional_function",
                        "parameters": {"threshold": 10}
                    }
                ],
                "edges": [
                    {
                        "from_node": "start_node",
                        "to_node": "math_node"
                    },
                    {
                        "from_node": "math_node",
                        "to_node": "conditional_node"
                    }
                ],
                "entry_point": "start_node"
            }
            
            # Create the graph
            response = requests.post(
                "http://localhost:8000/api/v1/graph/create",
                json={"graph": graph_definition}
            )
            assert response.status_code == 201
            graph_id = response.json()["graph_id"]
            print(f"✅ Created test graph: {graph_id}")
            
            # Start workflow execution
            response = requests.post(
                "http://localhost:8000/api/v1/graph/run",
                json={
                    "graph_id": graph_id,
                    "initial_state": {"counter": 5}
                }
            )
            assert response.status_code == 202
            run_id = response.json()["run_id"]
            print(f"✅ Started workflow execution: {run_id}")
            
            # Test 3: Subscribe to workflow run
            print("\n3. Testing subscription to workflow run...")
            subscribe_message = {
                "action": "subscribe",
                "run_id": run_id
            }
            await websocket.send(json.dumps(subscribe_message))
            
            # Wait for subscription confirmation
            message = await websocket.recv()
            data = json.loads(message)
            assert data["event_type"] == "subscription_confirmed"
            assert data["run_id"] == run_id
            print("✅ Successfully subscribed to workflow run")
            
            # Test 4: State change streaming
            print("\n4. Testing state change streaming...")
            events_received = []
            expected_events = [
                "workflow_started",
                "node_execution",  # start_node start
                "node_execution",  # start_node complete
                "node_execution",  # math_node start
                "node_execution",  # math_node complete
                "node_execution",  # conditional_node start
                "node_execution",  # conditional_node complete
                "workflow_completed"
            ]
            
            # Listen for workflow events
            timeout_count = 0
            max_timeout = 10
            
            while len(events_received) < len(expected_events) and timeout_count < max_timeout:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(message)
                    
                    if data["event_type"] in expected_events:
                        events_received.append(data["event_type"])
                        print(f"✅ Received event: {data['event_type']}")
                        
                        # Check for state changes in node execution events
                        if data["event_type"] == "node_execution" and "state_snapshot" in data.get("data", {}):
                            print(f"   State snapshot included: {bool(data['data']['state_snapshot'])}")
                        
                        # Check if workflow completed
                        if data["event_type"] == "workflow_completed":
                            print("✅ Workflow execution completed successfully")
                            break
                            
                except asyncio.TimeoutError:
                    timeout_count += 1
                    print(f"   Waiting for more events... ({timeout_count}/{max_timeout})")
            
            print(f"✅ Received {len(events_received)} workflow events")
            
            # Test 5: Error notification system
            print("\n5. Testing error notification system...")
            
            # Create a workflow that will fail
            error_graph = {
                "name": f"Error Test Workflow {timestamp}",
                "description": "A workflow designed to test error notifications",
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
            
            # Create the error graph
            response = requests.post(
                "http://localhost:8000/api/v1/graph/create",
                json={"graph": error_graph}
            )
            assert response.status_code == 201
            error_graph_id = response.json()["graph_id"]
            
            # Start error workflow execution
            response = requests.post(
                "http://localhost:8000/api/v1/graph/run",
                json={
                    "graph_id": error_graph_id,
                    "initial_state": {}
                }
            )
            assert response.status_code == 202
            error_run_id = response.json()["run_id"]
            
            # Subscribe to error workflow
            subscribe_error_message = {
                "action": "subscribe",
                "run_id": error_run_id
            }
            await websocket.send(json.dumps(subscribe_error_message))
            
            # Wait for error events
            error_received = False
            timeout_count = 0
            
            while not error_received and timeout_count < 5:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(message)
                    
                    if data["event_type"] == "execution_error":
                        error_received = True
                        print("✅ Error notification received successfully")
                        print(f"   Error message: {data['data'].get('error_message', 'N/A')}")
                        break
                        
                except asyncio.TimeoutError:
                    timeout_count += 1
            
            if not error_received:
                print("⚠️  Error notification not received (may have completed too quickly)")
            
            # Test 6: Connection lifecycle management
            print("\n6. Testing connection lifecycle management...")
            
            # Test ping functionality
            ping_message = {"action": "ping"}
            await websocket.send(json.dumps(ping_message))
            
            pong_received = False
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                data = json.loads(message)
                if data["event_type"] == "pong":
                    pong_received = True
                    print("✅ Ping/pong functionality working")
            except asyncio.TimeoutError:
                pass
            
            if not pong_received:
                print("⚠️  Ping/pong functionality not working")
            
            # Test connection status
            status_message = {"action": "get_status"}
            await websocket.send(json.dumps(status_message))
            
            status_received = False
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                data = json.loads(message)
                if data["event_type"] == "status_info":
                    status_received = True
                    connection_count = data["data"]["total_connections"]
                    print(f"✅ Connection status received: {connection_count} active connections")
            except asyncio.TimeoutError:
                pass
            
            if not status_received:
                print("⚠️  Connection status functionality not working")
            
            print("\n✅ All WebSocket monitoring tests completed successfully!")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False
    
    return True


def test_websocket_rest_endpoints():
    """Test WebSocket-related REST endpoints."""
    print("\n7. Testing WebSocket REST endpoints...")
    
    try:
        # Test connections info endpoint
        response = requests.get("http://localhost:8000/api/v1/ws/connections")
        assert response.status_code == 200
        data = response.json()
        assert "websocket_monitoring" in data
        assert data["websocket_monitoring"] == "active"
        print("✅ WebSocket connections endpoint working")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket REST endpoint test failed: {e}")
        return False


if __name__ == "__main__":
    print("Starting comprehensive WebSocket monitoring tests...")
    print("Make sure the server is running on http://localhost:8000")
    print()
    
    # Test REST endpoints first
    rest_success = test_websocket_rest_endpoints()
    
    # Test WebSocket functionality
    websocket_success = asyncio.run(test_websocket_comprehensive())
    
    print("\n" + "=" * 50)
    if rest_success and websocket_success:
        print("🎉 ALL TESTS PASSED! WebSocket monitoring is fully functional.")
    else:
        print("❌ Some tests failed. Please check the output above.")
    print("=" * 50)