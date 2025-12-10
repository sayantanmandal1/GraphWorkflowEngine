"""Simplified final integration tests for the complete workflow engine system."""

import pytest
import tempfile
import os
import time
import json
import threading
from datetime import datetime
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.tool_registry import ToolRegistry
from app.core.state_manager import StateManager
from app.core.graph_manager import GraphManager
from app.core.execution_engine import ExecutionEngine
from app.core.websocket_manager import WebSocketManager
from app.models.core import (
    GraphDefinition, NodeDefinition, EdgeDefinition, 
    ExecutionStatusEnum
)
from app.storage.database import create_tables


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)
    
    # Store original DATABASE_URL
    original_db_url = os.environ.get('DATABASE_URL')
    
    # Set database URL for testing
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
    
    # Create tables
    create_tables()
    
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
def workflow_engine(temp_db):
    """Create a complete workflow engine setup."""
    tool_registry = ToolRegistry()
    state_manager = StateManager()
    graph_manager = GraphManager()
    websocket_manager = WebSocketManager()
    execution_engine = ExecutionEngine(
        tool_registry=tool_registry,
        state_manager=state_manager,
        graph_manager=graph_manager,
        max_concurrent_executions=10,
        websocket_manager=websocket_manager
    )
    
    # Register test tools
    def simple_test_tool(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        message = kwargs.get('message', 'default')
        counter = state.get('counter', 0) + 1
        return {
            'message': f'Tool executed: {message}',
            'counter': counter,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def math_tool(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
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
        threshold = kwargs.get('threshold', 10)
        current_value = state.get('value', 0)
        
        return {
            'meets_threshold': current_value >= threshold,
            'checked_value': current_value,
            'threshold_used': threshold
        }
    
    # Register tools if not already registered
    tools_to_register = [
        ('simple_test_tool', simple_test_tool, 'Simple test tool'),
        ('math_tool', math_tool, 'Math operation tool'),
        ('conditional_tool', conditional_tool, 'Conditional logic tool')
    ]
    
    for tool_name, tool_func, description in tools_to_register:
        try:
            if not tool_registry.tool_exists(tool_name):
                tool_registry.register_tool(tool_name, tool_func, description)
        except Exception:
            pass  # Tool already exists
    
    yield {
        'tool_registry': tool_registry,
        'state_manager': state_manager,
        'graph_manager': graph_manager,
        'execution_engine': execution_engine,
        'websocket_manager': websocket_manager
    }
    
    # Cleanup
    execution_engine.shutdown()


class TestEndToEndWorkflowExecution:
    """Test complete end-to-end workflow execution."""
    
    def test_simple_workflow_execution(self, workflow_engine):
        """Test basic workflow creation and execution."""
        components = workflow_engine
        graph_manager = components['graph_manager']
        execution_engine = components['execution_engine']
        
        # Create a simple workflow
        workflow_def = GraphDefinition(
            name="Simple Integration Test",
            description="Basic end-to-end test",
            nodes=[
                NodeDefinition(
                    id="test_node",
                    function_name="simple_test_tool",
                    parameters={"message": "Hello from integration test"}
                )
            ],
            edges=[],
            entry_point="test_node"
        )
        
        # Create graph
        graph_id = graph_manager.create_graph(workflow_def)
        assert graph_id is not None
        
        # Execute workflow
        initial_state = {"test_value": "initial"}
        run_id = execution_engine.execute_workflow(graph_id, initial_state)
        assert run_id is not None
        
        # Wait for completion
        max_attempts = 30
        for attempt in range(max_attempts):
            status = execution_engine.get_execution_status(run_id)
            if status.status in [ExecutionStatusEnum.COMPLETED, ExecutionStatusEnum.FAILED]:
                break
            time.sleep(1)
        
        # Verify completion
        final_status = execution_engine.get_execution_status(run_id)
        assert final_status.status == ExecutionStatusEnum.COMPLETED
        assert "message" in final_status.current_state.data
        assert "counter" in final_status.current_state.data
        assert final_status.current_state.data["counter"] == 1
        
        # Get execution logs
        logs = execution_engine.get_execution_logs(run_id)
        assert len(logs) > 0
        
        # Verify logs are chronologically ordered
        log_timestamps = [log.timestamp for log in logs]
        assert log_timestamps == sorted(log_timestamps)
    
    def test_multi_step_workflow_execution(self, workflow_engine):
        """Test multi-step workflow with sequential execution."""
        components = workflow_engine
        graph_manager = components['graph_manager']
        execution_engine = components['execution_engine']
        
        # Create multi-step workflow
        workflow_def = GraphDefinition(
            name="Multi-Step Integration Test",
            description="Sequential execution test",
            nodes=[
                NodeDefinition(
                    id="init",
                    function_name="math_tool",
                    parameters={"operation": "add", "value": 5}
                ),
                NodeDefinition(
                    id="process",
                    function_name="math_tool",
                    parameters={"operation": "multiply", "value": 2}
                ),
                NodeDefinition(
                    id="finalize",
                    function_name="simple_test_tool",
                    parameters={"message": "Processing complete"}
                )
            ],
            edges=[
                EdgeDefinition(from_node="init", to_node="process"),
                EdgeDefinition(from_node="process", to_node="finalize")
            ],
            entry_point="init"
        )
        
        # Create and execute workflow
        graph_id = graph_manager.create_graph(workflow_def)
        run_id = execution_engine.execute_workflow(graph_id, {"result": 10})
        
        # Wait for completion
        max_attempts = 30
        for attempt in range(max_attempts):
            status = execution_engine.get_execution_status(run_id)
            if status.status in [ExecutionStatusEnum.COMPLETED, ExecutionStatusEnum.FAILED]:
                break
            time.sleep(1)
        
        # Verify final state
        final_status = execution_engine.get_execution_status(run_id)
        assert final_status.status == ExecutionStatusEnum.COMPLETED
        
        final_state = final_status.current_state.data
        # Should be: (10 + 5) * 2 = 30
        assert final_state["result"] == 30
        assert "message" in final_state  # From simple_test_tool
        assert "Processing complete" in final_state["message"]
    
    def test_conditional_branching_workflow(self, workflow_engine):
        """Test workflow with conditional branching logic."""
        components = workflow_engine
        graph_manager = components['graph_manager']
        execution_engine = components['execution_engine']
        
        # Create workflow with conditional branching
        workflow_def = GraphDefinition(
            name="Conditional Branching Test",
            description="Test conditional execution paths",
            nodes=[
                NodeDefinition(
                    id="check",
                    function_name="conditional_tool",
                    parameters={"threshold": 15}
                ),
                NodeDefinition(
                    id="high_path",
                    function_name="simple_test_tool",
                    parameters={"message": "High value path"}
                ),
                NodeDefinition(
                    id="low_path",
                    function_name="simple_test_tool",
                    parameters={"message": "Low value path"}
                )
            ],
            edges=[
                EdgeDefinition(
                    from_node="check",
                    to_node="high_path",
                    condition="state.get('meets_threshold', False) == True"
                ),
                EdgeDefinition(
                    from_node="check",
                    to_node="low_path",
                    condition="state.get('meets_threshold', False) == False"
                )
            ],
            entry_point="check"
        )
        
        # Test high value path
        graph_id = graph_manager.create_graph(workflow_def)
        run_id = execution_engine.execute_workflow(graph_id, {"value": 20})  # Above threshold
        
        # Wait for completion
        max_attempts = 30
        for attempt in range(max_attempts):
            status = execution_engine.get_execution_status(run_id)
            if status.status in [ExecutionStatusEnum.COMPLETED, ExecutionStatusEnum.FAILED]:
                break
            time.sleep(1)
        
        # Verify high path was taken
        final_status = execution_engine.get_execution_status(run_id)
        assert final_status.status == ExecutionStatusEnum.COMPLETED
        final_state = final_status.current_state.data
        assert "High value path" in final_state.get("message", "")


class TestConcurrentExecutionPerformance:
    """Test concurrent execution scenarios and performance."""
    
    def test_concurrent_workflow_isolation(self, workflow_engine):
        """Test that concurrent workflows maintain state isolation."""
        components = workflow_engine
        graph_manager = components['graph_manager']
        execution_engine = components['execution_engine']
        
        # Create a simple workflow
        workflow_def = GraphDefinition(
            name="Concurrent Isolation Test",
            description="Test concurrent execution isolation",
            nodes=[
                NodeDefinition(
                    id="process",
                    function_name="math_tool",
                    parameters={"operation": "add", "value": 1}
                )
            ],
            edges=[],
            entry_point="process"
        )
        
        graph_id = graph_manager.create_graph(workflow_def)
        
        # Start multiple concurrent executions with different initial states
        run_ids = []
        initial_values = [10, 20, 30, 40, 50]
        
        for value in initial_values:
            run_id = execution_engine.execute_workflow(graph_id, {"result": value})
            run_ids.append(run_id)
        
        # Wait for all executions to complete
        completed_states = {}
        max_attempts = 30
        
        for run_id in run_ids:
            for attempt in range(max_attempts):
                status = execution_engine.get_execution_status(run_id)
                if status.status in [ExecutionStatusEnum.COMPLETED, ExecutionStatusEnum.FAILED]:
                    completed_states[run_id] = status
                    break
                time.sleep(0.5)
        
        # Verify all executions completed successfully
        assert len(completed_states) == len(run_ids)
        
        # Verify state isolation - each should have its expected result
        results = []
        for run_id in run_ids:
            status = completed_states[run_id]
            assert status.status == ExecutionStatusEnum.COMPLETED
            result = status.current_state.data["result"]
            results.append(result)
        
        # Results should be [11, 21, 31, 41, 51] (each initial + 1)
        expected_results = [v + 1 for v in initial_values]
        assert sorted(results) == sorted(expected_results)
    
    def test_performance_under_load(self, workflow_engine):
        """Test system performance under concurrent load."""
        components = workflow_engine
        graph_manager = components['graph_manager']
        execution_engine = components['execution_engine']
        
        # Create a simple but realistic workflow
        workflow_def = GraphDefinition(
            name="Performance Load Test",
            description="Test performance under load",
            nodes=[
                NodeDefinition(
                    id="step1",
                    function_name="math_tool",
                    parameters={"operation": "add", "value": 2}
                ),
                NodeDefinition(
                    id="step2",
                    function_name="math_tool",
                    parameters={"operation": "multiply", "value": 3}
                )
            ],
            edges=[
                EdgeDefinition(from_node="step1", to_node="step2")
            ],
            entry_point="step1"
        )
        
        graph_id = graph_manager.create_graph(workflow_def)
        
        # Launch multiple concurrent executions
        num_executions = 10
        start_time = time.time()
        run_ids = []
        
        for i in range(num_executions):
            run_id = execution_engine.execute_workflow(graph_id, {"result": i})
            run_ids.append(run_id)
        
        # Wait for all to complete
        completed_count = 0
        max_wait_time = 60  # seconds
        
        while completed_count < num_executions and (time.time() - start_time) < max_wait_time:
            completed_count = 0
            
            for run_id in run_ids:
                status = execution_engine.get_execution_status(run_id)
                if status.status in [ExecutionStatusEnum.COMPLETED, ExecutionStatusEnum.FAILED, ExecutionStatusEnum.CANCELLED]:
                    completed_count += 1
            
            time.sleep(0.5)
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert completed_count == num_executions, f"Only {completed_count}/{num_executions} completed"
        assert total_time < 30, f"Execution took too long: {total_time:.2f}s"
        
        # Verify all completed successfully
        success_count = 0
        for run_id in run_ids:
            status = execution_engine.get_execution_status(run_id)
            if status.status == ExecutionStatusEnum.COMPLETED:
                success_count += 1
        
        assert success_count == num_executions, f"Only {success_count}/{num_executions} succeeded"


class TestSystemIntegrationWithDatabase:
    """Test system integration with real database operations."""
    
    def test_state_persistence_and_retrieval(self, workflow_engine):
        """Test that workflow state persists correctly in database."""
        components = workflow_engine
        graph_manager = components['graph_manager']
        execution_engine = components['execution_engine']
        
        # Create workflow
        workflow_def = GraphDefinition(
            name="Persistence Test Workflow",
            description="Test state persistence",
            nodes=[
                NodeDefinition(
                    id="persist_data",
                    function_name="simple_test_tool",
                    parameters={"message": "Persistent data"}
                )
            ],
            edges=[],
            entry_point="persist_data"
        )
        
        graph_id = graph_manager.create_graph(workflow_def)
        
        # Execute workflow
        initial_state = {"persistent_value": "test_data_123"}
        run_id = execution_engine.execute_workflow(graph_id, initial_state)
        
        # Wait for completion
        max_attempts = 30
        for attempt in range(max_attempts):
            status = execution_engine.get_execution_status(run_id)
            if status.status in [ExecutionStatusEnum.COMPLETED, ExecutionStatusEnum.FAILED]:
                break
            time.sleep(1)
        
        # Verify completion and state persistence
        final_status = execution_engine.get_execution_status(run_id)
        assert final_status.status == ExecutionStatusEnum.COMPLETED
        
        original_state = final_status.current_state.data
        assert original_state["persistent_value"] == "test_data_123"
        assert "message" in original_state
        
        # Test retrieval after some time (simulating system restart)
        time.sleep(2)
        
        # Query the same run_id again
        retrieved_status = execution_engine.get_execution_status(run_id)
        retrieved_state = retrieved_status.current_state.data
        
        # Verify state was persisted and can be retrieved
        assert retrieved_status.status == ExecutionStatusEnum.COMPLETED
        assert retrieved_state["persistent_value"] == original_state["persistent_value"]
        assert retrieved_state["message"] == original_state["message"]
    
    def test_historical_data_retrieval(self, workflow_engine):
        """Test retrieval of historical workflow execution data."""
        components = workflow_engine
        graph_manager = components['graph_manager']
        execution_engine = components['execution_engine']
        
        # Create workflow
        workflow_def = GraphDefinition(
            name="Historical Data Test",
            description="Test historical data retrieval",
            nodes=[
                NodeDefinition(
                    id="log_execution",
                    function_name="simple_test_tool",
                    parameters={"message": "Historical execution"}
                )
            ],
            edges=[],
            entry_point="log_execution"
        )
        
        graph_id = graph_manager.create_graph(workflow_def)
        
        # Execute multiple runs
        run_ids = []
        execution_times = []
        
        for i in range(3):
            start_time = datetime.utcnow()
            run_id = execution_engine.execute_workflow(graph_id, {"execution_number": i + 1})
            run_ids.append(run_id)
            execution_times.append(start_time)
            
            time.sleep(1)  # Ensure different timestamps
        
        # Wait for all executions to complete
        for run_id in run_ids:
            max_attempts = 30
            for attempt in range(max_attempts):
                status = execution_engine.get_execution_status(run_id)
                if status.status in [ExecutionStatusEnum.COMPLETED, ExecutionStatusEnum.FAILED]:
                    break
                time.sleep(1)
        
        # Test historical data retrieval
        for i, run_id in enumerate(run_ids):
            # Get execution status
            status = execution_engine.get_execution_status(run_id)
            
            assert status.status == ExecutionStatusEnum.COMPLETED
            assert status.current_state.data["execution_number"] == i + 1
            
            # Get execution logs
            logs = execution_engine.get_execution_logs(run_id)
            
            assert len(logs) > 0
            # Verify logs are chronologically ordered
            log_timestamps = [log.timestamp for log in logs]
            assert log_timestamps == sorted(log_timestamps)
    
    def test_error_handling_and_recovery(self, workflow_engine):
        """Test system error handling and recovery mechanisms."""
        components = workflow_engine
        graph_manager = components['graph_manager']
        execution_engine = components['execution_engine']
        
        # Create workflow that will cause an error
        workflow_def = GraphDefinition(
            name="Error Handling Test",
            description="Test error handling and recovery",
            nodes=[
                NodeDefinition(
                    id="error_node",
                    function_name="nonexistent_function",  # This will cause an error
                    parameters={}
                )
            ],
            edges=[],
            entry_point="error_node"
        )
        
        graph_id = graph_manager.create_graph(workflow_def)
        run_id = execution_engine.execute_workflow(graph_id, {"test": "error_handling"})
        
        # Wait for execution to fail
        max_attempts = 30
        final_status = None
        
        for attempt in range(max_attempts):
            status = execution_engine.get_execution_status(run_id)
            if status.status in [ExecutionStatusEnum.COMPLETED, ExecutionStatusEnum.FAILED, ExecutionStatusEnum.CANCELLED]:
                final_status = status
                break
            time.sleep(1)
        
        # Verify error was handled gracefully
        assert final_status is not None
        assert final_status.status == ExecutionStatusEnum.FAILED
        assert final_status.error_message is not None
        assert "nonexistent_function" in final_status.error_message
        
        # Verify system is still operational after error
        # Create and execute a valid workflow
        valid_workflow_def = GraphDefinition(
            name="Recovery Test Workflow",
            description="Test system recovery after error",
            nodes=[
                NodeDefinition(
                    id="recovery_node",
                    function_name="simple_test_tool",
                    parameters={"message": "System recovered"}
                )
            ],
            edges=[],
            entry_point="recovery_node"
        )
        
        recovery_graph_id = graph_manager.create_graph(valid_workflow_def)
        recovery_run_id = execution_engine.execute_workflow(recovery_graph_id, {"recovery_test": True})
        
        # Wait for recovery execution to complete
        for attempt in range(max_attempts):
            status = execution_engine.get_execution_status(recovery_run_id)
            if status.status == ExecutionStatusEnum.COMPLETED:
                break
            time.sleep(1)
        
        # Verify system recovered and can execute workflows normally
        recovery_status = execution_engine.get_execution_status(recovery_run_id)
        assert recovery_status.status == ExecutionStatusEnum.COMPLETED
        assert "System recovered" in recovery_status.current_state.data["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])