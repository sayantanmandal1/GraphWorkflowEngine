#!/usr/bin/env python3
"""
Comprehensive Workflow Engine Demo

This script demonstrates the full capabilities of the workflow engine including:
- Complex multi-stage data processing pipelines
- Conditional branching and parallel execution
- Error handling and recovery workflows
- Real-time monitoring via WebSocket
- State management and persistence
- Advanced workflow patterns
"""

import asyncio
import json
import time
import websockets
import requests
from datetime import datetime
from typing import Dict, Any, List
import threading
import queue


class WorkflowEngineDemo:
    """Comprehensive demonstration of workflow engine capabilities."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
        self.ws_url = f"ws://localhost:8000/api/v1/ws/monitor"
        self.event_queue = queue.Queue()
        
    def create_data_processing_pipeline(self) -> Dict[str, Any]:
        """Create a comprehensive data processing pipeline workflow."""
        return {
            "name": "Enterprise Data Processing Pipeline",
            "description": "Multi-stage data processing with validation, transformation, aggregation, and reporting",
            "nodes": [
                # Data Ingestion Stage
                {
                    "id": "data_ingestion",
                    "function_name": "test_function",
                    "parameters": {
                        "message": "Starting data ingestion from multiple sources",
                        "data": {
                            "source_systems": ["CRM", "ERP", "Analytics"],
                            "record_count": 50000,
                            "data_quality_score": 0.95
                        }
                    }
                },
                
                # Data Validation Stage
                {
                    "id": "data_validation",
                    "function_name": "conditional_function",
                    "parameters": {
                        "threshold": 0.9,
                        "condition_type": "quality_check"
                    }
                },
                
                # Parallel Processing Branches
                {
                    "id": "customer_data_processing",
                    "function_name": "simple_math_function",
                    "parameters": {
                        "operation": "multiply",
                        "value": 1.2,
                        "context": "customer_enrichment"
                    }
                },
                {
                    "id": "transaction_processing",
                    "function_name": "simple_math_function",
                    "parameters": {
                        "operation": "add",
                        "value": 100,
                        "context": "transaction_normalization"
                    }
                },
                {
                    "id": "analytics_processing",
                    "function_name": "simple_math_function",
                    "parameters": {
                        "operation": "multiply",
                        "value": 0.8,
                        "context": "analytics_aggregation"
                    }
                },
                
                # Data Aggregation
                {
                    "id": "data_aggregation",
                    "function_name": "simple_math_function",
                    "parameters": {
                        "operation": "add",
                        "value": 0,
                        "context": "final_aggregation"
                    }
                },
                
                # Quality Assessment
                {
                    "id": "quality_assessment",
                    "function_name": "conditional_function",
                    "parameters": {
                        "threshold": 1000,
                        "condition_type": "volume_check"
                    }
                },
                
                # Reporting and Logging
                {
                    "id": "generate_report",
                    "function_name": "state_logger_function",
                    "parameters": {
                        "log_level": "info",
                        "message": "Data processing pipeline completed - generating comprehensive report"
                    }
                },
                
                # Final Notification
                {
                    "id": "send_notification",
                    "function_name": "test_function",
                    "parameters": {
                        "message": "Pipeline execution completed successfully",
                        "notification_type": "success"
                    }
                }
            ],
            "edges": [
                # Sequential flow from ingestion to validation
                {"from_node": "data_ingestion", "to_node": "data_validation"},
                
                # Parallel processing branches
                {"from_node": "data_validation", "to_node": "customer_data_processing"},
                {"from_node": "data_validation", "to_node": "transaction_processing"},
                {"from_node": "data_validation", "to_node": "analytics_processing"},
                
                # Convergence to aggregation
                {"from_node": "customer_data_processing", "to_node": "data_aggregation"},
                {"from_node": "transaction_processing", "to_node": "data_aggregation"},
                {"from_node": "analytics_processing", "to_node": "data_aggregation"},
                
                # Quality check and reporting
                {"from_node": "data_aggregation", "to_node": "quality_assessment"},
                {"from_node": "quality_assessment", "to_node": "generate_report"},
                {"from_node": "generate_report", "to_node": "send_notification"}
            ],
            "entry_point": "data_ingestion"
        }
    
    def create_ml_training_pipeline(self) -> Dict[str, Any]:
        """Create a machine learning training pipeline workflow."""
        return {
            "name": "ML Model Training Pipeline",
            "description": "End-to-end machine learning pipeline with data prep, training, validation, and deployment",
            "nodes": [
                {
                    "id": "data_preparation",
                    "function_name": "test_function",
                    "parameters": {
                        "message": "Preparing training dataset",
                        "data": {
                            "dataset_size": 100000,
                            "features": 50,
                            "target_variable": "conversion_rate"
                        }
                    }
                },
                {
                    "id": "feature_engineering",
                    "function_name": "simple_math_function",
                    "parameters": {
                        "operation": "multiply",
                        "value": 1.5,
                        "context": "feature_scaling"
                    }
                },
                {
                    "id": "model_training",
                    "function_name": "simple_math_function",
                    "parameters": {
                        "operation": "add",
                        "value": 200,
                        "context": "training_iterations"
                    }
                },
                {
                    "id": "model_validation",
                    "function_name": "conditional_function",
                    "parameters": {
                        "threshold": 0.85,
                        "condition_type": "accuracy_check"
                    }
                },
                {
                    "id": "model_deployment",
                    "function_name": "state_logger_function",
                    "parameters": {
                        "log_level": "info",
                        "message": "Model deployed to production environment"
                    }
                }
            ],
            "edges": [
                {"from_node": "data_preparation", "to_node": "feature_engineering"},
                {"from_node": "feature_engineering", "to_node": "model_training"},
                {"from_node": "model_training", "to_node": "model_validation"},
                {"from_node": "model_validation", "to_node": "model_deployment"}
            ],
            "entry_point": "data_preparation"
        }
    
    def create_error_handling_workflow(self) -> Dict[str, Any]:
        """Create a workflow designed to demonstrate error handling."""
        return {
            "name": "Error Handling and Recovery Workflow",
            "description": "Demonstrates error handling, retry logic, and recovery mechanisms",
            "nodes": [
                {
                    "id": "initialize_process",
                    "function_name": "test_function",
                    "parameters": {
                        "message": "Initializing error-prone process",
                        "simulate_error": False
                    }
                },
                {
                    "id": "risky_operation",
                    "function_name": "nonexistent_function",  # This will cause an error
                    "parameters": {
                        "operation": "high_risk_calculation"
                    }
                },
                {
                    "id": "error_recovery",
                    "function_name": "state_logger_function",
                    "parameters": {
                        "log_level": "warning",
                        "message": "Error detected - initiating recovery procedure"
                    }
                }
            ],
            "edges": [
                {"from_node": "initialize_process", "to_node": "risky_operation"},
                {"from_node": "risky_operation", "to_node": "error_recovery"}
            ],
            "entry_point": "initialize_process"
        }
    
    async def monitor_workflow_websocket(self, run_id: str, duration: int = 60):
        """Monitor workflow execution via WebSocket."""
        print(f"\n🔌 Starting WebSocket monitoring for run: {run_id}")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Wait for connection established
                message = await websocket.recv()
                connection_data = json.loads(message)
                print(f"✅ WebSocket connected: {connection_data['connection_id']}")
                
                # Subscribe to the workflow run
                subscribe_msg = {
                    "action": "subscribe",
                    "run_id": run_id
                }
                await websocket.send(json.dumps(subscribe_msg))
                
                # Wait for subscription confirmation
                message = await websocket.recv()
                sub_data = json.loads(message)
                if sub_data["event_type"] == "subscription_confirmed":
                    print(f"✅ Subscribed to workflow run: {run_id}")
                
                # Monitor events
                start_time = time.time()
                event_count = 0
                
                while time.time() - start_time < duration:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        event_data = json.loads(message)
                        event_count += 1
                        
                        event_type = event_data.get("event_type")
                        timestamp = event_data.get("timestamp", "")
                        
                        if event_type == "workflow_started":
                            print(f"🚀 [{timestamp}] Workflow execution started")
                        elif event_type == "node_execution":
                            node_data = event_data.get("data", {})
                            node_id = node_data.get("node_id", "unknown")
                            execution_event = node_data.get("execution_event", "")
                            message_text = node_data.get("message", "")
                            print(f"⚙️  [{timestamp}] Node {node_id}: {execution_event} - {message_text}")
                        elif event_type == "execution_error":
                            error_data = event_data.get("data", {})
                            error_msg = error_data.get("error_message", "Unknown error")
                            print(f"❌ [{timestamp}] Execution error: {error_msg}")
                        elif event_type == "workflow_completed":
                            print(f"✅ [{timestamp}] Workflow completed successfully")
                            break
                        elif event_type == "execution_status_update":
                            status_data = event_data.get("data", {})
                            status = status_data.get("status", "unknown")
                            print(f"📊 [{timestamp}] Status update: {status}")
                        
                        # Store event for later analysis
                        self.event_queue.put(event_data)
                        
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        print("🔌 WebSocket connection closed")
                        break
                
                print(f"📊 WebSocket monitoring completed. Received {event_count} events.")
                
        except Exception as e:
            print(f"❌ WebSocket monitoring error: {str(e)}")
    
    def run_workflow(self, workflow_def: Dict[str, Any], initial_state: Dict[str, Any] = None) -> str:
        """Create and execute a workflow, returning the run ID."""
        if initial_state is None:
            initial_state = {}
        
        print(f"\n📝 Creating workflow: {workflow_def['name']}")
        
        # Create workflow
        response = requests.post(
            f"{self.api_url}/graph/create",
            json={"graph": workflow_def}
        )
        
        if response.status_code != 201:
            print(f"❌ Failed to create workflow: {response.text}")
            return None
        
        graph_data = response.json()
        graph_id = graph_data["graph_id"]
        print(f"✅ Workflow created: {graph_id}")
        
        # Start execution
        response = requests.post(
            f"{self.api_url}/graph/run",
            json={
                "graph_id": graph_id,
                "initial_state": initial_state
            }
        )
        
        if response.status_code != 202:
            print(f"❌ Failed to start workflow: {response.text}")
            return None
        
        run_data = response.json()
        run_id = run_data["run_id"]
        print(f"🏃 Workflow execution started: {run_id}")
        
        return run_id
    
    def wait_for_completion(self, run_id: str, timeout: int = 60) -> Dict[str, Any]:
        """Wait for workflow completion and return final status."""
        print(f"\n⏳ Waiting for workflow completion: {run_id}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = requests.get(f"{self.api_url}/graph/state/{run_id}")
            
            if response.status_code != 200:
                print(f"❌ Failed to get status: {response.text}")
                return None
            
            status_data = response.json()
            current_status = status_data["status"]
            
            if current_status in ["completed", "failed", "cancelled"]:
                print(f"🎯 Workflow {current_status}!")
                return status_data
            
            time.sleep(1)
        
        print("⏰ Timeout waiting for completion")
        return None
    
    def get_execution_logs(self, run_id: str) -> List[Dict[str, Any]]:
        """Retrieve and display execution logs."""
        print(f"\n📜 Retrieving execution logs for: {run_id}")
        
        response = requests.get(f"{self.api_url}/graph/logs/{run_id}")
        
        if response.status_code != 200:
            print(f"❌ Failed to get logs: {response.text}")
            return []
        
        logs = response.json()
        print(f"✅ Retrieved {len(logs)} log entries")
        
        # Display recent logs
        for log in logs[-15:]:
            timestamp = log["timestamp"]
            node_id = log["node_id"]
            event_type = log["event_type"]
            message = log["message"]
            print(f"   [{timestamp}] {node_id} - {event_type}: {message}")
        
        return logs
    
    def demonstrate_concurrent_workflows(self):
        """Demonstrate running multiple workflows concurrently."""
        print(f"\n🔄 Demonstrating concurrent workflow execution")
        
        # Create multiple workflows
        workflows = [
            (self.create_data_processing_pipeline(), {"batch_id": 1, "priority": "high"}),
            (self.create_ml_training_pipeline(), {"model_version": "v2.1", "dataset": "production"}),
            (self.create_data_processing_pipeline(), {"batch_id": 2, "priority": "normal"})
        ]
        
        run_ids = []
        
        # Start all workflows
        for workflow_def, initial_state in workflows:
            run_id = self.run_workflow(workflow_def, initial_state)
            if run_id:
                run_ids.append(run_id)
        
        print(f"\n🚀 Started {len(run_ids)} concurrent workflows")
        
        # Monitor all workflows
        completed_workflows = []
        start_time = time.time()
        timeout = 120
        
        while len(completed_workflows) < len(run_ids) and time.time() - start_time < timeout:
            for run_id in run_ids:
                if run_id not in completed_workflows:
                    response = requests.get(f"{self.api_url}/graph/state/{run_id}")
                    if response.status_code == 200:
                        status_data = response.json()
                        if status_data["status"] in ["completed", "failed", "cancelled"]:
                            completed_workflows.append(run_id)
                            print(f"✅ Workflow {run_id} completed with status: {status_data['status']}")
            
            time.sleep(2)
        
        print(f"\n📊 Concurrent execution summary:")
        print(f"   Total workflows: {len(run_ids)}")
        print(f"   Completed: {len(completed_workflows)}")
        print(f"   Execution time: {time.time() - start_time:.2f} seconds")
        
        return run_ids
    
    async def run_comprehensive_demo(self):
        """Run the complete comprehensive demo."""
        print("🚀 COMPREHENSIVE WORKFLOW ENGINE DEMONSTRATION")
        print("=" * 60)
        print("This demo showcases:")
        print("• Complex multi-stage data processing pipelines")
        print("• Machine learning workflow orchestration")
        print("• Error handling and recovery mechanisms")
        print("• Real-time monitoring via WebSocket")
        print("• Concurrent workflow execution")
        print("• State management and persistence")
        print("=" * 60)
        
        try:
            # Test server connectivity
            response = requests.get(f"{self.api_url}/graphs")
            if response.status_code != 200:
                print("❌ Server not accessible. Please ensure the workflow engine is running.")
                return
            
            print("✅ Server connectivity confirmed")
            
            # Demo 1: Data Processing Pipeline with WebSocket Monitoring
            print(f"\n" + "="*50)
            print("DEMO 1: DATA PROCESSING PIPELINE WITH REAL-TIME MONITORING")
            print("="*50)
            
            data_pipeline = self.create_data_processing_pipeline()
            initial_state = {
                "batch_size": 10000,
                "processing_mode": "optimized",
                "quality_threshold": 0.95,
                "source_systems": ["CRM", "ERP", "Analytics"]
            }
            
            run_id = self.run_workflow(data_pipeline, initial_state)
            if run_id:
                # Start WebSocket monitoring in background
                monitor_task = asyncio.create_task(
                    self.monitor_workflow_websocket(run_id, duration=60)
                )
                
                # Wait for completion
                final_status = self.wait_for_completion(run_id, timeout=60)
                
                # Get logs
                logs = self.get_execution_logs(run_id)
                
                # Wait for WebSocket monitoring to complete
                await monitor_task
                
                print(f"\n📊 Pipeline Results:")
                if final_status:
                    print(f"   Status: {final_status['status']}")
                    print(f"   Final State: {final_status.get('current_state', {})}")
            
            # Demo 2: Machine Learning Pipeline
            print(f"\n" + "="*50)
            print("DEMO 2: MACHINE LEARNING TRAINING PIPELINE")
            print("="*50)
            
            ml_pipeline = self.create_ml_training_pipeline()
            ml_initial_state = {
                "dataset_version": "v3.2",
                "model_type": "gradient_boosting",
                "hyperparameters": {
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "n_estimators": 100
                }
            }
            
            ml_run_id = self.run_workflow(ml_pipeline, ml_initial_state)
            if ml_run_id:
                ml_status = self.wait_for_completion(ml_run_id, timeout=30)
                ml_logs = self.get_execution_logs(ml_run_id)
                
                print(f"\n📊 ML Pipeline Results:")
                if ml_status:
                    print(f"   Status: {ml_status['status']}")
                    print(f"   Training completed: {ml_status.get('completed_at', 'N/A')}")
            
            # Demo 3: Error Handling
            print(f"\n" + "="*50)
            print("DEMO 3: ERROR HANDLING AND RECOVERY")
            print("="*50)
            
            error_workflow = self.create_error_handling_workflow()
            error_run_id = self.run_workflow(error_workflow, {"retry_count": 3})
            if error_run_id:
                error_status = self.wait_for_completion(error_run_id, timeout=30)
                error_logs = self.get_execution_logs(error_run_id)
                
                print(f"\n📊 Error Handling Results:")
                if error_status:
                    print(f"   Status: {error_status['status']}")
                    if error_status.get('error_message'):
                        print(f"   Error: {error_status['error_message']}")
            
            # Demo 4: Concurrent Workflows
            print(f"\n" + "="*50)
            print("DEMO 4: CONCURRENT WORKFLOW EXECUTION")
            print("="*50)
            
            concurrent_run_ids = self.demonstrate_concurrent_workflows()
            
            # Demo 5: System Overview
            print(f"\n" + "="*50)
            print("DEMO 5: SYSTEM OVERVIEW AND ANALYTICS")
            print("="*50)
            
            # List all workflows
            response = requests.get(f"{self.api_url}/graphs")
            if response.status_code == 200:
                graphs = response.json()
                print(f"\n📚 Total workflows in system: {len(graphs)}")
                for graph in graphs[-5:]:  # Show last 5
                    print(f"   • {graph['name']} (Nodes: {graph['node_count']})")
            
            # WebSocket connection info
            response = requests.get(f"{self.api_url}/ws/connections")
            if response.status_code == 200:
                ws_info = response.json()
                print(f"\n🔌 WebSocket Status: {ws_info.get('websocket_monitoring', 'Unknown')}")
                if 'connection_info' in ws_info:
                    conn_info = ws_info['connection_info']
                    print(f"   Active connections: {conn_info.get('total_connections', 0)}")
            
            # Event summary
            event_count = self.event_queue.qsize()
            print(f"\n📊 Real-time events captured: {event_count}")
            
            print(f"\n" + "="*60)
            print("🎉 COMPREHENSIVE DEMO COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("The workflow engine has demonstrated:")
            print("✅ Complex pipeline orchestration")
            print("✅ Real-time monitoring and event streaming")
            print("✅ Error handling and recovery")
            print("✅ Concurrent execution capabilities")
            print("✅ State management and persistence")
            print("✅ RESTful API and WebSocket integration")
            print("="*60)
            
        except requests.exceptions.ConnectionError:
            print("❌ Connection error: Please ensure the workflow engine server is running on http://localhost:8000")
        except Exception as e:
            print(f"❌ Demo error: {str(e)}")


def main():
    """Main entry point for the comprehensive demo."""
    demo = WorkflowEngineDemo()
    
    print("Starting Comprehensive Workflow Engine Demo...")
    print("Please ensure the server is running: uvicorn app.main:app --port 8000")
    print()
    
    # Run the async demo
    asyncio.run(demo.run_comprehensive_demo())


if __name__ == "__main__":
    main()