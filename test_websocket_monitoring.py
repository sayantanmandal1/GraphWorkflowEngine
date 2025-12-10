#!/usr/bin/env python3
"""Test script for WebSocket real-time monitoring functionality."""

import asyncio
import json
import websockets
import requests
from datetime import datetime


async def test_websocket_monitoring():
    """Test WebSocket monitoring functionality."""
    print("Testing WebSocket real-time monitoring...")
    
    # First, create a simple workflow graph
    import time
    timestamp = int(time.time())
    graph_definition = {
        "name": f"WebSocket Test Workflow {timestamp}",
        "description": "A simple workflow to test WebSocket monitoring",
        "nodes": [
            {
                "id": "start_node",
                "function_name": "test_function",
                "parameters": {"message": "Starting WebSocket test"}
            },
            {
                "id": "math_node", 
                "function_name": "simple_math_function",
                "parameters": {"operation": "add", "value": 10}
            }
        ],
        "edges": [
            {
                "from_node": "start_node",
                "to_node": "math_node"
            }
        ],
        "entry_point": "start_node"
    }
    
    # Create the graph
    print("Creating workflow graph...")
    response = requests.post(
        "http://localhost:8000/api/v1/graph/create",
        json={"graph": graph_definition}
    )
    
    if response.status_code != 201:
        print(f"Failed to create graph: {response.status_code} - {response.text}")
        return
    
    graph_id = response.json()["graph_id"]
    print(f"Created graph with ID: {graph_id}")
    
    # Connect to WebSocket
    print("Connecting to WebSocket...")
    uri = "ws://localhost:8000/api/v1/ws/monitor"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("WebSocket connected successfully!")
            
            # Wait for connection established message
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Received: {data}")
            
            # Start workflow execution
            print("Starting workflow execution...")
            response = requests.post(
                "http://localhost:8000/api/v1/graph/run",
                json={
                    "graph_id": graph_id,
                    "initial_state": {"counter": 5}
                }
            )
            
            if response.status_code != 202:
                print(f"Failed to start workflow: {response.status_code} - {response.text}")
                return
            
            run_id = response.json()["run_id"]
            print(f"Started workflow with run ID: {run_id}")
            
            # Subscribe to workflow events
            subscribe_message = {
                "action": "subscribe",
                "run_id": run_id
            }
            await websocket.send(json.dumps(subscribe_message))
            print(f"Subscribed to run {run_id}")
            
            # Listen for events
            print("Listening for WebSocket events...")
            event_count = 0
            max_events = 10  # Limit to prevent infinite loop
            
            while event_count < max_events:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(message)
                    event_count += 1
                    
                    print(f"Event {event_count}: {data['event_type']}")
                    if 'run_id' in data:
                        print(f"  Run ID: {data['run_id']}")
                    if 'message' in data.get('data', {}):
                        print(f"  Message: {data['data']['message']}")
                    if data['event_type'] in ['workflow_completed', 'workflow_cancelled', 'execution_error']:
                        print("Workflow execution finished")
                        break
                        
                except asyncio.TimeoutError:
                    print("Timeout waiting for events")
                    break
                except Exception as e:
                    print(f"Error receiving message: {e}")
                    break
            
            # Test ping
            print("Testing ping...")
            ping_message = {"action": "ping"}
            await websocket.send(json.dumps(ping_message))
            
            try:
                pong_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                pong_data = json.loads(pong_response)
                print(f"Ping response: {pong_data}")
            except asyncio.TimeoutError:
                print("No pong response received")
            
            # Test connection status
            print("Getting connection status...")
            status_message = {"action": "get_status"}
            await websocket.send(json.dumps(status_message))
            
            try:
                status_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                status_data = json.loads(status_response)
                print(f"Connection status: {status_data}")
            except asyncio.TimeoutError:
                print("No status response received")
            
            print("WebSocket test completed successfully!")
            
    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"WebSocket error: {e}")


def test_websocket_connections_endpoint():
    """Test the WebSocket connections info endpoint."""
    print("\nTesting WebSocket connections endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/api/v1/ws/connections")
        if response.status_code == 200:
            data = response.json()
            print(f"WebSocket connections info: {json.dumps(data, indent=2)}")
        else:
            print(f"Failed to get connections info: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error testing connections endpoint: {e}")


if __name__ == "__main__":
    print("WebSocket Monitoring Test")
    print("=" * 50)
    print("Make sure the server is running on http://localhost:8000")
    print()
    
    # Test REST endpoint first
    test_websocket_connections_endpoint()
    
    # Test WebSocket functionality
    asyncio.run(test_websocket_monitoring())